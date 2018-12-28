package YaraParser.Accessories;

import YaraParser.Learning.AveragedPerceptron;
import YaraParser.Learning.BinaryPerceptron;
import YaraParser.Structures.CompactArray;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.InfStruct;
import YaraParser.Structures.Sentence;
import YaraParser.TransitionBasedSystem.Configuration.*;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import YaraParser.TransitionBasedSystem.Parser.Actions;
import YaraParser.TransitionBasedSystem.Parser.ArcEager;
import YaraParser.TransitionBasedSystem.Parser.BeamScorerThread;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class BinaryModelEvaluator {
    private Options options;
    private AveragedPerceptron classifier;                          //maybe no needed
    private BinaryPerceptron bClassifier;
    private String model;
    private ArrayList<Integer> dependencyRelations;
    private int featureLength;
    private int all;
    private float match;
    private Random randGen;

    public BinaryModelEvaluator(String modelFile, AveragedPerceptron classifier, BinaryPerceptron bClassifier,
            Options options, ArrayList<Integer> dependencyRelations, int featureLength) {
        model = modelFile;
        this.classifier = classifier;
        this.bClassifier = bClassifier;
        this.options = options;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        randGen = new Random();
    }

    public void evaluate() throws Exception {
        /**
         * Actions: 0=shift, 1=reduce, 2=unshift, ra_dep=3+dep,
         * la_dep=3+dependencyRelations.size()+dep
         */
        ExecutorService executor = Executors.newFixedThreadPool(options.numOfThreads);
        CompletionService<ArrayList<BeamElement>> pool = new ExecutorCompletionService<ArrayList<BeamElement>>(
                executor);

        CoNLLReader goldReader = new CoNLLReader(options.inputFile);
        IndexMaps maps = CoNLLReader.createIndices(options.inputFile, options.labeled, options.lowercase,
                options.clusterFile);
        ArrayList<GoldConfiguration> trainData = goldReader.readData(Integer.MAX_VALUE, false, options.labeled,
                options.rootFirst, options.lowercase, maps);
        for (int i = 1; i <= options.trainingIter; i++) {
            long start = System.currentTimeMillis();
            System.out.println("### BinaryModelEvaluator:");
            int dataCount = 0;
            int progress = Math.max(trainData.size() / 100, 100 / trainData.size());
            int percentage = 0;
            all = 0;
            match = 0f;
            System.out.println("train size " + trainData.size());
            System.out.print("progress: " + percentage++ + "%\r");
            for (GoldConfiguration goldConfiguration : trainData) {
                dataCount++;
                if (dataCount % progress == 0)
                    System.out.print("progress: " + percentage++ + "%\r");
                System.out.println("Entering trainOnOneSample");
                trainOnOneSample(goldConfiguration, i, dataCount, pool);
                System.out.println("Come Out of trainOnOneSample");
                classifier.incrementIteration();
                bClassifier.incrementIteration();
            }
            System.out.print("\n");
            System.out.println("train phase completed!");
            long end = System.currentTimeMillis();
            long timeSec = (end - start) / 1000;
            System.out.println("iteration " + i + " took " + timeSec + " seconds\n");
            DecimalFormat format = new DecimalFormat("##.00");
            System.err.println("Accuracy: " + format.format(100.0 * match / all));
            System.out.println("done\n");
        }
        boolean isTerminated = executor.isTerminated();
        while (!isTerminated) {
            executor.shutdownNow();
            isTerminated = executor.isTerminated();
        }
    }

    private void trainOnOneSample(GoldConfiguration goldConfiguration, int i, int dataCount,
            CompletionService<ArrayList<BeamElement>> pool) throws Exception {
        boolean isPartial = goldConfiguration.isPartial(options.rootFirst);

        if (options.partialTrainingStartingIteration > i && isPartial)
            return;

        Sentence sentence = goldConfiguration.getSentence();

        Configuration initialConfiguration = new Configuration(goldConfiguration.getSentence(), options.rootFirst);
        Configuration firstOracle = initialConfiguration.clone();
        ArrayList<Configuration> beam = new ArrayList<Configuration>(options.beamWidth);
        beam.add(initialConfiguration);

        /**
         * The float is the oracle's cost For more information see: Yoav Goldberg and
         * Joakim Nivre. "Training Deterministic Parsers with Non-Deterministic
         * Oracles." TACL 1 (2013): 403-414. for the mean while we just use zero-cost
         * oracles
         */
        HashMap<Configuration, Float> oracles = new HashMap<Configuration, Float>();

        oracles.put(firstOracle, 0.0f);

        /**
         * For keeping track of the violations For more information see: Liang Huang,
         * Suphan Fayong and Yang Guo. "Structured perceptron with inexact search." In
         * Proceedings of the 2012 Conference of the North American Chapter of the
         * Association for Computational Linguistics: Human Language Technologies, pp.
         * 142-151. Association for Computational Linguistics, 2012.
         */
        float maxViol = Float.NEGATIVE_INFINITY;
        Pair<Configuration, Configuration> maxViolPair = null;

        Configuration bestScoringOracle = null;

        while (!ArcEager.isTerminal(beam) && beam.size() > 0) {
            /**
             * generating new oracles it keeps the oracles which are in the terminal state
             */
            HashMap<Configuration, Float> newOracles = new HashMap<Configuration, Float>();

            if (options.useDynamicOracle || isPartial) {
                bestScoringOracle = zeroCostDynamicOracle(goldConfiguration, oracles, newOracles);
            } else {
                bestScoringOracle = staticOracle(goldConfiguration, oracles, newOracles);
            }

            if (newOracles.size() == 0) {
                System.err.print("...no oracle(" + dataCount + ")...");
            }
            oracles = newOracles;

            TreeSet<BeamElement> beamPreserver = new TreeSet<BeamElement>();

            if (options.numOfThreads == 1 || beam.size() == 1) {
                beamSortOneThread(beam, beamPreserver, sentence);
            } else {
                for (int b = 0; b < beam.size(); b++) {
                    pool.submit(new BeamScorerThread(false, classifier, beam.get(b), dependencyRelations, featureLength,
                            b, options.rootFirst));
                }
                for (int b = 0; b < beam.size(); b++) {
                    for (BeamElement element : pool.take().get()) {
                        beamPreserver.add(element);
                        if (beamPreserver.size() > options.beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }

            if (beamPreserver.size() == 0 || beam.size() == 0) {
                break;
            } else {
                ArrayList<Configuration> repBeam = new ArrayList<Configuration>(options.beamWidth);
                for (BeamElement beamElement : beamPreserver.descendingSet()) {
                    if (repBeam.size() >= options.beamWidth)
                        break;
                    int b = beamElement.number;
                    int action = beamElement.action;
                    int label = beamElement.label;
                    float sc = beamElement.score;

                    Configuration newConfig = beam.get(b).clone();

                    if (action == 0) {
                        ArcEager.shift(newConfig.state);
                        newConfig.addAction(0);
                    } else if (action == 1) {
                        ArcEager.reduce(newConfig.state);
                        newConfig.addAction(1);
                    } else if (action == 2) {
                        ArcEager.rightArc(newConfig.state, label);
                        newConfig.addAction(3 + label);
                    } else if (action == 3) {
                        ArcEager.leftArc(newConfig.state, label);
                        newConfig.addAction(3 + dependencyRelations.size() + label);
                    } else if (action == 4) {
                        ArcEager.unShift(newConfig.state);
                        newConfig.addAction(2);
                    }
                    newConfig.setScore(sc);
                    repBeam.add(newConfig);
                    all++;
                    // Binary classifier update
                    if (oracles.containsKey(newConfig) == isOracle(newConfig, label))
                        match++;
                    System.out.println("all: " + all + "match: " + match);
                }
                beam = repBeam;

                if (beam.size() > 0 && oracles.size() > 0) {
                    Configuration bestConfig = beam.get(0);
                    if (oracles.containsKey(bestConfig)) {
                        oracles = new HashMap<Configuration, Float>();
                        oracles.put(bestConfig, 0.0f);
                    } else {
                        if (options.useRandomOracleSelection) { // choosing randomly, otherwise using latent structured
                                                                // Perceptron
                            List<Configuration> keys = new ArrayList<Configuration>(oracles.keySet());
                            Configuration randomKey = keys.get(randGen.nextInt(keys.size()));
                            oracles = new HashMap<Configuration, Float>();
                            oracles.put(randomKey, 0.0f);
                            bestScoringOracle = randomKey;
                        } else {
                            oracles = new HashMap<Configuration, Float>();
                            oracles.put(bestScoringOracle, 0.0f);
                        }
                    }
                } else
                    break;
            }
        }
    }

    private Configuration staticOracle(GoldConfiguration goldConfiguration, HashMap<Configuration, Float> oracles,
            HashMap<Configuration, Float> newOracles) throws Exception {
        Configuration bestScoringOracle = null;
        int top = -1;
        int first = -1;
        HashMap<Integer, Pair<Integer, Integer>> goldDependencies = goldConfiguration.getGoldDependencies();
        HashMap<Integer, HashSet<Integer>> reversedDependencies = goldConfiguration.getReversedDependencies();

        for (Configuration configuration : oracles.keySet()) {
            State state = configuration.state;
            Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);

            if (!state.stackEmpty())
                top = state.peek();
            if (!state.bufferEmpty())
                first = state.bufferHead();

            if (!configuration.state.isTerminalState()) {
                Configuration newConfig = configuration.clone();

                if (first > 0 && goldDependencies.containsKey(first) && goldDependencies.get(first).first == top) {
                    int dependency = goldDependencies.get(first).second;
                    float[] scores = classifier.rightArcScores(features, false);
                    float score = scores[dependency];
                    ArcEager.rightArc(newConfig.state, dependency);
                    newConfig.addAction(3 + dependency);
                    newConfig.addScore(score);
                } else if (top > 0 && goldDependencies.containsKey(top) && goldDependencies.get(top).first == first) {
                    int dependency = goldDependencies.get(top).second;
                    float[] scores = classifier.leftArcScores(features, false);
                    float score = scores[dependency];
                    ArcEager.leftArc(newConfig.state, dependency);
                    newConfig.addAction(3 + dependencyRelations.size() + dependency);
                    newConfig.addScore(score);
                } else if (top >= 0 && state.hasHead(top)) {

                    if (reversedDependencies.containsKey(top)) {
                        if (reversedDependencies.get(top).size() == state.valence(top)) {
                            float score = classifier.reduceScore(features, false);
                            ArcEager.reduce(newConfig.state);
                            newConfig.addAction(1);
                            newConfig.addScore(score);
                        } else {
                            float score = classifier.shiftScore(features, false);
                            ArcEager.shift(newConfig.state);
                            newConfig.addAction(0);
                            newConfig.addScore(score);
                        }
                    } else {
                        float score = classifier.reduceScore(features, false);
                        ArcEager.reduce(newConfig.state);
                        newConfig.addAction(1);
                        newConfig.addScore(score);
                    }

                } else if (state.bufferEmpty() && state.stackSize() == 1 && state.peek() == state.rootIndex) {
                    float score = classifier.reduceScore(features, false);
                    ArcEager.reduce(newConfig.state);
                    newConfig.addAction(1);
                    newConfig.addScore(score);
                } else {
                    float score = classifier.shiftScore(features, true);
                    ArcEager.shift(newConfig.state);
                    newConfig.addAction(0);
                    newConfig.addScore(score);
                }
                bestScoringOracle = newConfig;
                newOracles.put(newConfig, (float) 0);
            } else {
                newOracles.put(configuration, oracles.get(configuration));
            }
        }
        return bestScoringOracle;
    }

    private Configuration zeroCostDynamicOracle(GoldConfiguration goldConfiguration,
            HashMap<Configuration, Float> oracles, HashMap<Configuration, Float> newOracles) throws Exception {
        float bestScore = Float.NEGATIVE_INFINITY;
        Configuration bestScoringOracle = null;

        for (Configuration configuration : oracles.keySet()) {
            if (!configuration.state.isTerminalState()) {
                State currentState = configuration.state;
                Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
                int accepted = 0;
                // I only assumed that we need zero cost ones
                if (goldConfiguration.actionCost(Actions.Shift, -1, currentState) == 0) {
                    Configuration newConfig = configuration.clone();
                    float score = classifier.shiftScore(features, false);
                    ArcEager.shift(newConfig.state);
                    newConfig.addAction(0);
                    newConfig.addScore(score);
                    newOracles.put(newConfig, (float) 0);

                    if (newConfig.getScore(true) > bestScore) {
                        bestScore = newConfig.getScore(true);
                        bestScoringOracle = newConfig;
                    }
                    accepted++;
                }
                if (ArcEager.canDo(Actions.RightArc, currentState)) {
                    float[] rightArcScores = classifier.rightArcScores(features, false);
                    for (int dependency : dependencyRelations) {
                        if (goldConfiguration.actionCost(Actions.RightArc, dependency, currentState) == 0) {
                            Configuration newConfig = configuration.clone();
                            float score = rightArcScores[dependency];
                            ArcEager.rightArc(newConfig.state, dependency);
                            newConfig.addAction(3 + dependency);
                            newConfig.addScore(score);
                            newOracles.put(newConfig, (float) 0);

                            if (newConfig.getScore(true) > bestScore) {
                                bestScore = newConfig.getScore(true);
                                bestScoringOracle = newConfig;
                            }
                            accepted++;
                        }
                    }
                }
                if (ArcEager.canDo(Actions.LeftArc, currentState)) {
                    float[] leftArcScores = classifier.leftArcScores(features, false);

                    for (int dependency : dependencyRelations) {
                        if (goldConfiguration.actionCost(Actions.LeftArc, dependency, currentState) == 0) {
                            Configuration newConfig = configuration.clone();
                            float score = leftArcScores[dependency];
                            ArcEager.leftArc(newConfig.state, dependency);
                            newConfig.addAction(3 + dependencyRelations.size() + dependency);
                            newConfig.addScore(score);
                            newOracles.put(newConfig, (float) 0);

                            if (newConfig.getScore(true) > bestScore) {
                                bestScore = newConfig.getScore(true);
                                bestScoringOracle = newConfig;
                            }
                            accepted++;
                        }
                    }
                }
                if (goldConfiguration.actionCost(Actions.Reduce, -1, currentState) == 0) {
                    Configuration newConfig = configuration.clone();
                    float score = classifier.reduceScore(features, false);
                    ArcEager.reduce(newConfig.state);
                    newConfig.addAction(1);
                    newConfig.addScore(score);
                    newOracles.put(newConfig, (float) 0);

                    if (newConfig.getScore(true) > bestScore) {
                        bestScore = newConfig.getScore(true);
                        bestScoringOracle = newConfig;
                    }
                    accepted++;
                }
            } else {
                newOracles.put(configuration, oracles.get(configuration));
            }
        }

        return bestScoringOracle;
    }

    private void beamSortOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver, Sentence sentence)
            throws Exception {
        for (int b = 0; b < beam.size(); b++) {
            Configuration configuration = beam.get(b);
            State currentState = configuration.state;
            float prevScore = configuration.score;
            boolean canShift = ArcEager.canDo(Actions.Shift, currentState);
            boolean canReduce = ArcEager.canDo(Actions.Reduce, currentState);
            boolean canRightArc = ArcEager.canDo(Actions.RightArc, currentState);
            boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, currentState);
            Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);

            if (canShift) {
                float score = classifier.shiftScore(features, false);
                float addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 0, -1));

                if (beamPreserver.size() > options.beamWidth)
                    beamPreserver.pollFirst();
            }
            if (canReduce) {
                float score = classifier.reduceScore(features, false);
                float addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 1, -1));

                if (beamPreserver.size() > options.beamWidth)
                    beamPreserver.pollFirst();
            }

            if (canRightArc) {
                float[] rightArcScores = classifier.rightArcScores(features, false);
                for (int dependency : dependencyRelations) {
                    float score = rightArcScores[dependency];
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));

                    if (beamPreserver.size() > options.beamWidth)
                        beamPreserver.pollFirst();
                }
            }
            if (canLeftArc) {
                float[] leftArcScores = classifier.leftArcScores(features, false);
                for (int dependency : dependencyRelations) {
                    float score = leftArcScores[dependency];
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));

                    if (beamPreserver.size() > options.beamWidth)
                        beamPreserver.pollFirst();
                }
            }
        }
    }

    private boolean isOracle(Configuration bestConfiguration, int label) throws Exception {
        InfStruct infStruct = new InfStruct(model);
        /*dependencyLabels = (ArrayList<Integer>) reader.readObject();
        maps = (IndexMaps) reader.readObject();
        options = (Options) reader.readObject();
        shiftFeatureAveragedWeights = (HashMap<Object, Float>[]) reader.readObject();
        reduceFeatureAveragedWeights = (HashMap<Object, Float>[]) reader.readObject();
        leftArcFeatureAveragedWeights = (HashMap<Object, CompactArray>[]) reader.readObject();
        rightArcFeatureAveragedWeights = (HashMap<Object, CompactArray>[]) reader.readObject();
        dependencySize = reader.readInt();*/
//        State currentState = bestConfiguration.state;
//        float prevScore = bestConfiguration.score;
        int lastAction = bestConfiguration.actionHistory.get(bestConfiguration.actionHistory.size() - 1);
        Object[] features = FeatureExtractor.extractAllParseFeatures(bestConfiguration, featureLength);

        float score = 0.0f;
        //float score;
        if (lastAction == 0) {

            for (int i = 0; i < features.length; i++) {
                if (features[i] == null || (i >= 26 && i < 32))
                    continue;
                Float values = infStruct.shiftFeatureAveragedWeights[i].get(features[i]);
                if (values != null) {
                    score += values;
                }
            }
            //score = bClassifier.shiftScore(features, false);                    //shiftFeatureAveragedWeights--->>>infStruct    readObject
        } else if (lastAction == 1) {
            for (int i = 0; i < features.length; i++) {
                if (features[i] == null || (i >= 26 && i < 32))
                    continue;
                Float values = infStruct.reduceFeatureAveragedWeights[i].get(features[i]);
                if (values != null) {
                    score += values;
                }
            }
            //score = bClassifier.reduceScore(features, false);
        }

        else if ((lastAction - 3 - label) == 0) {
            float scores[] = new float[infStruct.dependencySize];
            for (int i = 0; i < features.length; i++) {
                if (features[i] == null)
                    continue;
                CompactArray values = infStruct.rightArcFeatureAveragedWeights[i].get(features[i]);
                if (values != null) {
                    int offset = values.getOffset();
                    float[] weightVector = values.getArray();

                    for (int d = offset; d < offset + weightVector.length; d++) {
                        scores[d] += weightVector[d - offset];
                    }
                }
            }
            //float[] rightArcScores = bClassifier.rightArcScores(features, false);
            score = scores[label];
        } else {
            float scores[] = new float[infStruct.dependencySize];
            for (int i = 0; i < features.length; i++) {
                if (features[i] == null)
                    continue;
                CompactArray values = infStruct.leftArcFeatureAveragedWeights[i].get(features[i]);
                if (values != null) {
                    int offset = values.getOffset();
                    float[] weightVector = values.getArray();

                    for (int d = offset; d < offset + weightVector.length; d++) {
                        scores[d] += weightVector[d - offset];
                    }
                }
            }
//            float[] leftArcScores = bClassifier.leftArcScores(features, false);
            score = scores[label];
        }
        return (score >= 0);
    }
}