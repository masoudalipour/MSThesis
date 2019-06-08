package YaraParser.Accessories;

import YaraParser.Learning.AveragedPerceptron;
import YaraParser.Learning.BinaryPerceptron;
import YaraParser.Structures.CompactArray;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.InfStruct;
import YaraParser.Structures.Sentence;
import YaraParser.TransitionBasedSystem.Configuration.BeamElement;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Configuration.State;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import YaraParser.TransitionBasedSystem.Parser.Actions;
import YaraParser.TransitionBasedSystem.Parser.ArcEager;
import YaraParser.TransitionBasedSystem.Parser.BeamScorerThread;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class BinaryModelEvaluator {
    private Options options;
    private AveragedPerceptron classifier; // maybe no needed
    private BinaryPerceptron bClassifier;
    private ArrayList<Integer> dependencyRelations;
    private int featureLength;
    private int TP;
    private int FP;
    private int TN;
    private int FN;
    private Random randGen;
    private InfStruct infStruct;

    public BinaryModelEvaluator(String modelFile, AveragedPerceptron classifier, BinaryPerceptron bClassifier,
                                Options options, ArrayList<Integer> dependencyRelations, int featureLength) throws Exception {
        this.classifier = classifier;
        this.bClassifier = bClassifier;
        this.options = options;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        randGen = new Random();
        infStruct = new InfStruct(modelFile);
    }

    public void evaluate() throws Exception {
        /**
         * Actions: 0=shift, 1=reduce, 2=unshift, ra_dep=3+dep,
         * la_dep=3+dependencyRelations.size()+dep
         */
        ExecutorService executor = Executors.newFixedThreadPool(options.numOfThreads);
        CompletionService<ArrayList<BeamElement>> pool = new ExecutorCompletionService<>(
                executor);
        CoNLLReader goldReader = new CoNLLReader(options.devPath);
        IndexMaps maps = CoNLLReader.createIndices(options.devPath, options.labeled, options.lowercase,
                options.clusterFile);
        ArrayList<GoldConfiguration> trainData = goldReader.readData(Integer.MAX_VALUE, false, options.labeled,
                options.rootFirst, options.lowercase, maps);
        long start = System.currentTimeMillis();
        System.out.println("### BinaryModelEvaluator:");
        int dataCount = 0;
        int progress = trainData.size() / 100;
        if (progress == 0) {
            progress = 1;
        }
        TP = 0;
        FP = 0;
        TN = 0;
        FN = 0;
        System.out.println("train size " + trainData.size());
        System.out.print("progress: 0%\r");
        for (GoldConfiguration goldConfiguration : trainData) {
            dataCount++;
            if (dataCount % progress == 0)
                System.out.print("progress: " + (dataCount * 100) / trainData.size() + "%\r");
            trainOnOneSample(goldConfiguration, dataCount, pool);
            classifier.incrementIteration();
            bClassifier.incrementIteration();
        }
        System.out.print("\n");
        System.out.println("train phase completed!");
        long end = System.currentTimeMillis();
        double timeSec = (double) (end - start) / 1000;
        DecimalFormat percentageFormat = new DecimalFormat("0.00%");
        percentageFormat.setRoundingMode(RoundingMode.HALF_UP);
        DecimalFormat decimalFormat = new DecimalFormat("0.000");
        decimalFormat.setRoundingMode(RoundingMode.HALF_UP);
        System.out.println("The evaluation took " + decimalFormat.format(timeSec) + " seconds\n");
        double accuracy = (double) (TP + TN) / (TP + TN + FP + FN);
        double precision = (double) TP / (TP + FP);
        double recall = (double) TP / (TP + FN);
        double f1Score = 2 * (recall * precision) / (recall + precision);
        double TN_TNFN = (double) TN / (TN + FN);
        double TN_TNFP = (double) TN / (TN + FP);
        System.out.println("TP: " + TP);
        System.out.println("FP: " + FP);
        System.out.println("TN: " + TN);
        System.out.println("FN: " + FN);
        System.out.println("Accuracy: " + percentageFormat.format(accuracy));
        System.out.println("Precision: " + decimalFormat.format(precision));
        System.out.println("Recall: " + decimalFormat.format(recall));
        System.out.println("F1 Score: " + decimalFormat.format(f1Score));
        System.out.println("TN / (TN + FN): " + percentageFormat.format(TN_TNFN));
        System.out.println("TN / (TN + FP): " + percentageFormat.format(TN_TNFP));
        System.out.println("done\n");
        boolean isTerminated = executor.isTerminated();
        while (!isTerminated) {
            executor.shutdownNow();
            isTerminated = executor.isTerminated();
        }
    }

    private void trainOnOneSample(GoldConfiguration goldConfiguration, int dataCount,
                                  CompletionService<ArrayList<BeamElement>> pool) throws Exception {
        boolean isPartial = goldConfiguration.isPartial(options.rootFirst);
        Sentence sentence = goldConfiguration.getSentence();
        Configuration initialConfiguration = new Configuration(goldConfiguration.getSentence(), options.rootFirst);
        Configuration firstOracle = initialConfiguration.clone();
        ArrayList<Configuration> beam = new ArrayList<>(options.beamWidth);
        beam.add(initialConfiguration);
        /**
         * The float is the oracle's cost For more information see: Yoav Goldberg and
         * Joakim Nivre. "Training Deterministic Parsers with Non-Deterministic
         * Oracles." TACL 1 (2013): 403-414. for the mean while we just use zero-cost
         * oracles
         */
        HashMap<Configuration, Float> oracles = new HashMap<>();
        oracles.put(firstOracle, 0.0f);
        /**
         * For keeping track of the violations For more information see: Liang Huang,
         * Suphan Fayong and Yang Guo. "Structured perceptron with inexact search." In
         * Proceedings of the 2012 Conference of the North American Chapter of the
         * Association for Computational Linguistics: Human Language Technologies, pp.
         * 142-151. Association for Computational Linguistics, 2012.
         */
        Configuration bestScoringOracle;
        while (ArcEager.isNotTerminal(beam) && beam.size() > 0) {
            /*
              generating new oracles it keeps the oracles which are in the terminal state
             */
            HashMap<Configuration, Float> newOracles = new HashMap<>();
            if (options.useDynamicOracle || isPartial) {
                bestScoringOracle = zeroCostDynamicOracle(goldConfiguration, oracles, newOracles);
            } else {
                bestScoringOracle = staticOracle(goldConfiguration, oracles, newOracles);
            }
            if (newOracles.size() == 0) {
                System.err.print("...no oracle(" + dataCount + ")...");
            }
            oracles = newOracles;
            TreeSet<BeamElement> beamPreserver = new TreeSet<>();
            if (options.numOfThreads == 1 || beam.size() == 1) {
                beamSortOneThread(beam, beamPreserver, sentence);
            } else {
                for (int b = 0; b < beam.size(); b++) {
                    pool.submit(new BeamScorerThread(false, classifier, beam.get(b), dependencyRelations, featureLength,
                            b));
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
                ArrayList<Configuration> repBeam = new ArrayList<>(options.beamWidth);
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
                    // Binary classifier update
                    boolean oracle = oracles.containsKey(newConfig);
                    boolean prediction = isOracle(newConfig, label);
                    if (oracle) {
                        if (prediction)
                            TP++;
                        else
                            FN++;
                    } else {
                        if (prediction)
                            FP++;
                        else
                            TN++;
                    }
                }
                beam = repBeam;
                if (beam.size() > 0 && oracles.size() > 0) {
                    Configuration bestConfig = beam.get(0);
                    if (oracles.containsKey(bestConfig)) {
                        oracles = new HashMap<>();
                        oracles.put(bestConfig, 0.0f);
                    } else {
                        if (options.useRandomOracleSelection) { // choosing randomly, otherwise using latent structured
                            // Perceptron
                            List<Configuration> keys = new ArrayList<>(oracles.keySet());
                            bestScoringOracle = keys.get(randGen.nextInt(keys.size()));
                        }
                        oracles = new HashMap<>();
                        oracles.put(bestScoringOracle, 0.0f);
                    }
                } else
                    break;
            }
        }
    }

    private Configuration staticOracle(GoldConfiguration goldConfiguration, HashMap<Configuration, Float> oracles,
                                       HashMap<Configuration, Float> newOracles) {
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
            if (configuration.state.isNotTerminalState()) {
                Configuration newConfig = configuration.clone();
                if (first > 0 && goldDependencies.containsKey(first) && goldDependencies.get(first).first == top) {
                    int dependency = goldDependencies.get(first).second;
                    float[] scores = classifier.rightArcScores(features, true);
                    float score = scores[dependency];
                    ArcEager.rightArc(newConfig.state, dependency);
                    newConfig.addAction(3 + dependency);
                    newConfig.addScore(score);
                } else if (top > 0 && goldDependencies.containsKey(top) && goldDependencies.get(top).first == first) {
                    int dependency = goldDependencies.get(top).second;
                    float[] scores = classifier.leftArcScores(features, true);
                    float score = scores[dependency];
                    ArcEager.leftArc(newConfig.state, dependency);
                    newConfig.addAction(3 + dependencyRelations.size() + dependency);
                    newConfig.addScore(score);
                } else if (top >= 0 && state.hasHead(top)) {
                    if (reversedDependencies.containsKey(top)) {
                        if (reversedDependencies.get(top).size() == state.valence(top)) {
                            float score = classifier.reduceScore(features, true);
                            ArcEager.reduce(newConfig.state);
                            newConfig.addAction(1);
                            newConfig.addScore(score);
                        } else {
                            float score = classifier.shiftScore(features, true);
                            ArcEager.shift(newConfig.state);
                            newConfig.addAction(0);
                            newConfig.addScore(score);
                        }
                    } else {
                        float score = classifier.reduceScore(features, true);
                        ArcEager.reduce(newConfig.state);
                        newConfig.addAction(1);
                        newConfig.addScore(score);
                    }
                } else if (state.bufferEmpty() && state.stackSize() == 1 && state.peek() == state.rootIndex) {
                    float score = classifier.reduceScore(features, true);
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
                                                HashMap<Configuration, Float> oracles,
                                                HashMap<Configuration, Float> newOracles) {
        float bestScore = Float.NEGATIVE_INFINITY;
        Configuration bestScoringOracle = null;
        for (Configuration configuration : oracles.keySet()) {
            if (configuration.state.isNotTerminalState()) {
                State currentState = configuration.state;
                Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
                // I only assumed that we need zero cost ones
                if (goldConfiguration.actionCost(Actions.Shift, -1, currentState) == 0) {
                    Configuration newConfig = configuration.clone();
                    float score = classifier.shiftScore(features, true);
                    ArcEager.shift(newConfig.state);
                    newConfig.addAction(0);
                    newConfig.addScore(score);
                    newOracles.put(newConfig, (float) 0);
                    if (newConfig.getScore() > bestScore) {
                        bestScore = newConfig.getScore();
                        bestScoringOracle = newConfig;
                    }
                }
                if (ArcEager.canDo(Actions.RightArc, currentState)) {
                    float[] rightArcScores = classifier.rightArcScores(features, true);
                    for (int dependency : dependencyRelations) {
                        if (goldConfiguration.actionCost(Actions.RightArc, dependency, currentState) == 0) {
                            Configuration newConfig = configuration.clone();
                            float score = rightArcScores[dependency];
                            ArcEager.rightArc(newConfig.state, dependency);
                            newConfig.addAction(3 + dependency);
                            newConfig.addScore(score);
                            newOracles.put(newConfig, (float) 0);
                            if (newConfig.getScore() > bestScore) {
                                bestScore = newConfig.getScore();
                                bestScoringOracle = newConfig;
                            }
                        }
                    }
                }
                if (ArcEager.canDo(Actions.LeftArc, currentState)) {
                    float[] leftArcScores = classifier.leftArcScores(features, true);
                    for (int dependency : dependencyRelations) {
                        if (goldConfiguration.actionCost(Actions.LeftArc, dependency, currentState) == 0) {
                            Configuration newConfig = configuration.clone();
                            float score = leftArcScores[dependency];
                            ArcEager.leftArc(newConfig.state, dependency);
                            newConfig.addAction(3 + dependencyRelations.size() + dependency);
                            newConfig.addScore(score);
                            newOracles.put(newConfig, (float) 0);
                            if (newConfig.getScore() > bestScore) {
                                bestScore = newConfig.getScore();
                                bestScoringOracle = newConfig;
                            }
                        }
                    }
                }
                if (goldConfiguration.actionCost(Actions.Reduce, -1, currentState) == 0) {
                    Configuration newConfig = configuration.clone();
                    float score = classifier.reduceScore(features, true);
                    ArcEager.reduce(newConfig.state);
                    newConfig.addAction(1);
                    newConfig.addScore(score);
                    newOracles.put(newConfig, (float) 0);
                    if (newConfig.getScore() > bestScore) {
                        bestScore = newConfig.getScore();
                        bestScoringOracle = newConfig;
                    }
                }
            } else {
                newOracles.put(configuration, oracles.get(configuration));
            }
        }
        return bestScoringOracle;
    }

    private void beamSortOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver,
                                   Sentence sentence) {
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
                float score = classifier.shiftScore(features, true);
                float addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 0, -1));
                if (beamPreserver.size() > options.beamWidth)
                    beamPreserver.pollFirst();
            }
            if (canReduce) {
                float score = classifier.reduceScore(features, true);
                float addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 1, -1));
                if (beamPreserver.size() > options.beamWidth)
                    beamPreserver.pollFirst();
            }
            if (canRightArc) {
                float[] rightArcScores = classifier.rightArcScores(features, true);
                for (int dependency : dependencyRelations) {
                    float score = rightArcScores[dependency];
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));
                    if (beamPreserver.size() > options.beamWidth)
                        beamPreserver.pollFirst();
                }
            }
            if (canLeftArc) {
                float[] leftArcScores = classifier.leftArcScores(features, true);
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

    private boolean isOracle(Configuration bestConfiguration, int label) {
        int lastAction = bestConfiguration.actionHistory.get(bestConfiguration.actionHistory.size() - 1);
        Object[] features = FeatureExtractor.extractAllParseFeatures(bestConfiguration, featureLength);
        float score = 0.0f;
        if (lastAction == 0) {
            for (int i = 0; i < features.length; i++) {
                if (features[i] == null || (i >= 26 && i < 32))
                    continue;
                Float values = infStruct.shiftFeatureAveragedWeights[i].get(features[i]);
                if (values != null) {
                    score += values;
                }
            }
        } else if (lastAction == 1) {
            for (int i = 0; i < features.length; i++) {
                if (features[i] == null || (i >= 26 && i < 32))
                    continue;
                Float values = infStruct.reduceFeatureAveragedWeights[i].get(features[i]);
                if (values != null) {
                    score += values;
                }
            }
        } else if ((lastAction - 3 - label) == 0) {
            float[] scores = new float[infStruct.dependencySize];
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
            score = scores[label];
        } else {
            float[] scores = new float[infStruct.dependencySize];
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
            score = scores[label];
        }
        return (score >= 0);
    }
}
