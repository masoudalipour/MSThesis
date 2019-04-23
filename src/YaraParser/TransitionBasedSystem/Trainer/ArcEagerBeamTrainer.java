package YaraParser.TransitionBasedSystem.Trainer;

import YaraParser.Accessories.BinaryModelEvaluator;
import YaraParser.Accessories.Evaluator;
import YaraParser.Accessories.Options;
import YaraParser.Accessories.Pair;
import YaraParser.Learning.AveragedPerceptron;
import YaraParser.Learning.BinaryPerceptron;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.InfStruct;
import YaraParser.TransitionBasedSystem.Configuration.BeamElement;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Configuration.State;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import YaraParser.TransitionBasedSystem.Parser.Actions;
import YaraParser.TransitionBasedSystem.Parser.ArcEager;
import YaraParser.TransitionBasedSystem.Parser.BeamScorerThread;
import YaraParser.TransitionBasedSystem.Parser.KBeamArcEagerParser;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ArcEagerBeamTrainer {
    private final int featureLength;
    private Options options;
    /**
     * Can be either "early" or "max_violation" For more information read: Liang
     * Huang, Suphan Fayong and Yang Guo. "Structured perceptron with inexact
     * search." In Proceedings of the 2012 Conference of the North American Chapter
     * of the Association for Computational Linguistics: Human Language
     * Technologies, pp. 142-151. Association for Computational Linguistics, 2012.
     */
    private String updateMode;
    private AveragedPerceptron classifier;
    private BinaryPerceptron bClassifier;
    private ArrayList<Integer> dependencyRelations;
    private Random randGen;
    private IndexMaps maps;

    public ArcEagerBeamTrainer(String updateMode, AveragedPerceptron classifier, Options options,
                               ArrayList<Integer> dependencyRelations, int featureLength, IndexMaps maps) {
        this.updateMode = updateMode;
        this.classifier = classifier;
        this.options = options;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        randGen = new Random();
        this.maps = maps;
    }

    public ArcEagerBeamTrainer(String updateMode, AveragedPerceptron classifier, BinaryPerceptron bClassifier,
                               Options options, ArrayList<Integer> dependencyRelations, int featureLength,
                               IndexMaps maps) {
        this.updateMode = updateMode;
        this.classifier = classifier;
        this.bClassifier = bClassifier;
        this.options = options;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        randGen = new Random();
        this.maps = maps;
    }

    public void train(ArrayList<GoldConfiguration> trainData, String devPath, int maxIteration, String modelPath,
                      boolean lowerCased, HashSet<String> punctuations, int partialTreeIter) throws Exception {
        /*
          Actions: 0=shift, 1=reduce, 2=unshift, ra_dep=3+dep,
          la_dep=3+dependencyRelations.size()+dep
         */
        ExecutorService executor = Executors.newFixedThreadPool(options.numOfThreads);
        CompletionService<ArrayList<BeamElement>> pool = new ExecutorCompletionService<>(executor);
        final int trainSize = trainData.size();
        for (int i = 1; i <= maxIteration; i++) {
            long start = System.currentTimeMillis();
            long startInNanos = System.nanoTime();
            System.out.println("### ArcEagerBeamTrainer:");
            int dataCount = 0;
            double progress = 1.0;
            if (trainSize > 100) {
                progress = (double) trainSize / 100;
            }
            if (trainSize < 100) {
                progress = (double) 100 / trainSize;
            }
            System.out.println("train size " + trainSize);
            System.out.print("progress: 0%\r");
            for (GoldConfiguration goldConfiguration : trainData) {
                dataCount++;
                if ((int) (dataCount % progress) == 0)
                    System.out.print("progress: " + (dataCount * 100) / trainSize + "%\r");
                trainOnOneSample(goldConfiguration, partialTreeIter, i, dataCount, pool);

                classifier.incrementIteration();
                bClassifier.incrementIteration();
            }
            System.out.print("\n");
            System.out.println("train phase completed!");
            long end = System.currentTimeMillis();
            long endInNanos = System.nanoTime();
            Duration duration = Duration.ofNanos(endInNanos - startInNanos);
            System.out.println("iteration " + i + " took " + duration.toString()
                    .substring(2)
                    .replaceAll("(\\d[HMS])(?!$)", "$1 ")
                    .toLowerCase());
            long timeSec = (end - start) / 1000;
            long timeMiliSec = (end - start) % 1000;
            System.out.println("iteration " + i + " took " + timeSec + "." + timeMiliSec + " seconds\n");

            System.out.println("Saving the model");
            if (modelPath.lastIndexOf("/") > 0) {
                String modelFolder = modelPath.substring(0, modelPath.lastIndexOf("/"));
                File modelDirectory = new File(modelFolder);
                if (!modelDirectory.exists()) {
                    if (!modelDirectory.mkdirs()) {
                        throw new Exception("Creating model's directory failed");
                    }

                }
            }
            InfStruct infStruct = new InfStruct(classifier, maps, dependencyRelations, options);
            InfStruct bInfStruct = new InfStruct(bClassifier, maps, dependencyRelations, options);
            infStruct.saveModel(modelPath + "_iter" + i);
            bInfStruct.saveModel(modelPath + "_Binary_iter" + i);
            System.out.println("The model saved");
            if (!devPath.equals("")) {
                System.out.println("Validating AveragedPerceptron model:");
                AveragedPerceptron averagedPerceptron = new AveragedPerceptron(infStruct);

                int raSize = averagedPerceptron.raSize();
                int effectiveRaSize = averagedPerceptron.effectiveRaSize();
                float raRatio = 100.0f * effectiveRaSize / raSize;
                int laSize = averagedPerceptron.laSize();
                int effectiveLaSize = averagedPerceptron.effectiveLaSize();
                float laRatio = 100.0f * effectiveLaSize / laSize;

                DecimalFormat format = new DecimalFormat("##.00");
                System.out.println("size of RA features in memory:" + effectiveRaSize + "/" + raSize + "->"
                        + format.format(raRatio) + "%");
                System.out.println("size of LA features in memory:" + effectiveLaSize + "/" + laSize + "->"
                        + format.format(laRatio) + "%");
                KBeamArcEagerParser parser = new KBeamArcEagerParser(averagedPerceptron, dependencyRelations,
                        featureLength, maps, options.numOfThreads);

                parser.parseCoNLLFile(devPath, modelPath + ".__tmp__", options.rootFirst, options.beamWidth, true,
                        lowerCased, options.numOfThreads, false, "");
                Evaluator.evaluate(devPath, modelPath + ".__tmp__", punctuations);
                Files.deleteIfExists(Path.of(modelPath + ".__tmp__"));

                parser.shutDownLiveThreads();
                System.out.println("Validating BinaryPerceptron model:");
                BinaryPerceptron binaryPerceptron = new BinaryPerceptron(bInfStruct);

                raSize = binaryPerceptron.raSize();
                effectiveRaSize = binaryPerceptron.effectiveRaSize();
                raRatio = 100.0f * effectiveRaSize / raSize;
                laSize = binaryPerceptron.laSize();
                effectiveLaSize = binaryPerceptron.effectiveLaSize();
                laRatio = 100.0f * effectiveLaSize / laSize;

                format = new DecimalFormat("##.00");
                System.out.println("size of RA features in memory:" + effectiveRaSize + "/" + raSize + "->"
                        + format.format(raRatio) + "%");
                System.out.println("size of LA features in memory:" + effectiveLaSize + "/" + laSize + "->"
                        + format.format(laRatio) + "%");

                BinaryModelEvaluator bEval = new BinaryModelEvaluator(modelPath + "_Binary_iter" + i, classifier,
                        bClassifier, options, dependencyRelations, featureLength);
                bEval.evaluate();
            }
        }
        boolean isTerminated = executor.isTerminated();
        while (!isTerminated) {
            executor.shutdownNow();
            isTerminated = executor.isTerminated();
        }
    }

    private void trainOnOneSample(GoldConfiguration goldConfiguration, int partialTreeIter, int i, int dataCount,
                                  CompletionService<ArrayList<BeamElement>> pool) throws Exception {
        boolean isPartial = goldConfiguration.isPartial(options.rootFirst);

        if (isPartial && partialTreeIter > i)
            return;

        Configuration initialConfiguration = new Configuration(goldConfiguration.getSentence(), options.rootFirst);
        Configuration firstOracle = initialConfiguration.clone();
        ArrayList<Configuration> beam = new ArrayList<>(options.beamWidth);
        beam.add(initialConfiguration);

        /*
          The float is the oracle's cost For more information see: Yoav Goldberg and
          Joakim Nivre. "Training Deterministic Parsers with Non-Deterministic
          Oracles." TACL 1 (2013): 403-414. for the mean while we just use zero-cost
          oracles
         */
        HashMap<Configuration, Float> oracles = new HashMap<>();

        oracles.put(firstOracle, 0.0f);

        /*
          For keeping track of the violations For more information see: Liang Huang,
          Suphan Fayong and Yang Guo. "Structured perceptron with inexact search." In
          Proceedings of the 2012 Conference of the North American Chapter of the
          Association for Computational Linguistics: Human Language Technologies, pp.
          142-151. Association for Computational Linguistics, 2012.
         */
        float maxViol = Float.NEGATIVE_INFINITY;

        Configuration bestScoringOracle = zeroCostDynamicOracle(goldConfiguration, oracles, new HashMap<>());
        Pair<Configuration, Configuration> maxViolPair = new Pair<>(beam.get(0), bestScoringOracle);
        boolean oracleInBeam = false;

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
                beamSortOneThread(beam, beamPreserver);
            } else {
                for (int b = 0; b < beam.size(); b++) {
                    pool.submit(new BeamScorerThread(false, classifier, beam.get(b), dependencyRelations, featureLength,
                            b, options.rootFirst));
                }
                /*
                  store top configurations as much as beam width in beamPreserver
                 */
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
                oracleInBeam = false;
                ArrayList<Configuration> repBeam = new ArrayList<>(options.beamWidth);
                for (BeamElement beamElement : beamPreserver.descendingSet()) {
                    if (repBeam.size() >= options.beamWidth)
                        break;

                    int b = beamElement.number;
                    int action = beamElement.action;
                    int label = beamElement.label;
                    float sc = beamElement.score;
                    Configuration newConfig = beam.get(b).clone();

                    switch (action) {
                        case 0:
                            ArcEager.shift(newConfig.state);
                            newConfig.addAction(0);
                            break;
                        case 1:
                            ArcEager.reduce(newConfig.state);
                            newConfig.addAction(1);
                            break;
                        case 2:
                            ArcEager.rightArc(newConfig.state, label);
                            newConfig.addAction(3 + label);
                            break;
                        case 3:
                            ArcEager.leftArc(newConfig.state, label);
                            newConfig.addAction(3 + dependencyRelations.size() + label);
                            break;
                        case 4:
                            ArcEager.unShift(newConfig.state);
                            newConfig.addAction(2);
                            break;
                    }
                    newConfig.setScore(sc);
                    repBeam.add(newConfig);

                    /*
                      Binary classifier update
                     */
                    if (oracles.containsKey(newConfig) != isOracle(newConfig))
                        updateWeights(true, initialConfiguration, isPartial, bestScoringOracle, newConfig);

                    if (oracles.containsKey(newConfig))
                        oracleInBeam = true;
                }
                beam = repBeam;

                if (beam.size() > 0 && oracles.size() > 0) {
                    Configuration bestConfig = beam.get(0);
                    if (oracles.containsKey(bestConfig)) {
                        oracles = new HashMap<>();
                        oracles.put(bestConfig, 0.0f);
                    } else {
                        if (options.useRandomOracleSelection) {
                            List<Configuration> keys = new ArrayList<>(oracles.keySet());
                            Configuration randomKey = keys.get(randGen.nextInt(keys.size()));
                            oracles = new HashMap<>();
                            oracles.put(randomKey, 0.0f);
                            bestScoringOracle = randomKey;
                        } else {
                            /*
                              latent structured Perceptron
                             */
                            oracles = new HashMap<>();
                            oracles.put(bestScoringOracle, 0.0f);
                        }
                    }

                    if (!oracleInBeam && updateMode.equals("early"))
                        break;

                    if (beam.size() > 0 && !oracleInBeam && updateMode.equals("max_violation")) {
                        float violation = beam.get(0).getScore() - bestScoringOracle.getScore();
                        if (violation > maxViol) {
                            maxViol = violation;
                            maxViolPair = new Pair<>(beam.get(0), bestScoringOracle);
                        }
                    }
                } else
                    break;
            }
        }

        /*
          updating weights
         */
        if (oracleInBeam && bestScoringOracle.equals(beam.get(0))) {
            return;
        }
        Configuration predicted;
        Configuration finalOracle;
        if (!updateMode.equals("max_violation")) {
            finalOracle = bestScoringOracle;
            predicted = beam.get(0);
        } else {
            float violation = beam.get(0).getScore() - bestScoringOracle.getScore();
            if (violation > maxViol) {
                maxViolPair = new Pair<>(beam.get(0), bestScoringOracle);
            }
            predicted = maxViolPair.first;
            finalOracle = maxViolPair.second;
        }
        updateWeights(false, initialConfiguration, isPartial, finalOracle, predicted);
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
                                                HashMap<Configuration, Float> oracles,
                                                HashMap<Configuration, Float> newOracles) {
        float bestScore = Float.NEGATIVE_INFINITY;
        Configuration bestScoringOracle = null;

        for (Configuration configuration : oracles.keySet()) {
            if (configuration.state.isNotTerminalState()) {
                State currentState = configuration.state;
                Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
                /*
                  It's assumed that we need actions that cost non
                 */
                if (goldConfiguration.actionCost(Actions.Shift, -1, currentState) == 0) {
                    Configuration newConfig = configuration.clone();
                    float score = classifier.shiftScore(features, false);
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
                    float[] rightArcScores = classifier.rightArcScores(features, false);
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
                    float[] leftArcScores = classifier.leftArcScores(features, false);

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
                    float score = classifier.reduceScore(features, false);
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

    private void beamSortOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver) {
        for (int b = 0; b < beam.size(); b++) {
            Configuration configuration = beam.get(b);
            State currentState = configuration.state;
            float prevScore = configuration.score;
            Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);

            if (ArcEager.canDo(Actions.Shift, currentState)) {
                float score = classifier.shiftScore(features, false);
                float addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 0, -1));

                if (beamPreserver.size() > options.beamWidth)
                    beamPreserver.pollFirst();
            }
            if (ArcEager.canDo(Actions.Reduce, currentState)) {
                float score = classifier.reduceScore(features, false);
                float addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 1, -1));

                if (beamPreserver.size() > options.beamWidth)
                    beamPreserver.pollFirst();
            }

            if (ArcEager.canDo(Actions.RightArc, currentState)) {
                float[] rightArcScores = classifier.rightArcScores(features, false);
                for (int dependency : dependencyRelations) {
                    float score = rightArcScores[dependency];
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));

                    if (beamPreserver.size() > options.beamWidth)
                        beamPreserver.pollFirst();
                }
            }
            if (ArcEager.canDo(Actions.LeftArc, currentState)) {
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

    private boolean checkIfTrueFeature(Configuration conf, boolean isPartial, int action) {
        if (isPartial) {
            if (action >= 3) {
                return conf.state.hasHead(conf.state.peek())
                        && conf.state.hasHead(conf.state.bufferHead());
            } else if (action == 0) {
                return conf.state.hasHead(conf.state.bufferHead());
            } else if (action == 1) {
                return conf.state.hasHead(conf.state.peek());
            }
        }
        return true;
    }

    /**
     * As it parses the initial configuration, it weights the list of features
     *
     * @param initialConfiguration The initial configuration that the parsing starts with
     * @param featuresValue        The list of hash-maps that contains features and their weight
     * @param actionHistory        The list of actions that leads the initial configuration to a parse tree
     * @param isPartial            Determines if the input is partial
     */
    private void weightFeatures(Configuration initialConfiguration,
                                List<HashMap<Pair<Integer, Object>, Float>> featuresValue,
                                ArrayList<Integer> actionHistory, boolean isPartial) {
        for (int action : actionHistory) {
            boolean isTrueFeature = checkIfTrueFeature(initialConfiguration, isPartial, action);
            if (isTrueFeature) { // if the made dependency is truly for the word
                Object[] feats = FeatureExtractor.extractAllParseFeatures(initialConfiguration, featureLength);
                for (int f = 0; f < feats.length; f++) {
                    Pair<Integer, Object> featName = new Pair<>(action, feats[f]);
                    HashMap<Pair<Integer, Object>, Float> map = featuresValue.get(f);
                    if (map.containsKey(featName)) {
                        Float value = map.get(featName);
                        map.put(featName, value + 1);
                    } else {
                        map.put(featName, 1.0f);
                    }
                }
            }
            State state = initialConfiguration.state;
            if (action == 0) {
                ArcEager.shift(state);
            } else if (action == 1) {
                ArcEager.reduce(state);
            } else if (action == 2) {
                ArcEager.unShift(state);
            } else if (action >= 3 + dependencyRelations.size()) {
                int dependency = action - (3 + dependencyRelations.size());
                ArcEager.leftArc(state, dependency);
            } else if (action >= 3) {
                int dependency = action - 3;
                ArcEager.rightArc(state, dependency);
            }
        }
    }

    /**
     * Determines the location of the action in
     *
     * @param index                     The index of dependency in the array of dependencies
     * @param action                    The action's type
     * @param dependencyRelationsLength The length of the array of dependencies
     * @return The index of dependency in its own category
     */
    private int getDependencyInsideIndex(int index, Actions action, int dependencyRelationsLength) {
        int dependencyIndex;
        switch (action) {
            case LeftArc:
                dependencyIndex = index - (3 + dependencyRelationsLength);
                break;
            case RightArc:
                dependencyIndex = index - 3;
                break;
            default:
                dependencyIndex = 0;
        }
        return dependencyIndex;
    }

    /**
     * Implements averaged structured perceptron
     *
     * @param isBinary             Determines if it's the binary classifier that is getting updated or the yara
     *                             parser classifier
     * @param initialConfiguration The initial configuration that the parsing starts with
     * @param isPartial            Determines if the input is partial
     * @param finalOracle          The oracle configuration
     * @param predicted            The predicted configuration
     */
    private void updateWeights(boolean isBinary, Configuration initialConfiguration, boolean isPartial,
                               Configuration finalOracle, Configuration predicted) {
        List<HashMap<Pair<Integer, Object>, Float>> oracleFeatures = new ArrayList<>();
        List<HashMap<Pair<Integer, Object>, Float>> predictedFeatures = new ArrayList<>();
        for (int f = 0; f < featureLength; f++) {
            oracleFeatures.add(new HashMap<>());
            predictedFeatures.add(new HashMap<>());
        }
        weightFeatures(initialConfiguration.clone(), oracleFeatures, finalOracle.actionHistory, isPartial);
        weightFeatures(initialConfiguration.clone(), predictedFeatures, predicted.actionHistory, isPartial);
        for (int f = 0; f < featureLength; f++) {
            HashMap<Pair<Integer, Object>, Float> predictedMap = predictedFeatures.get(f);
            HashMap<Pair<Integer, Object>, Float> oracleMap = oracleFeatures.get(f);
            for (Pair<Integer, Object> feat : predictedMap.keySet()) {
                if (feat.second != null) {
                    int action = feat.first;
                    Object feature = feat.second;
                    Actions actionType = Actions.intToAction(action, dependencyRelations.size());
                    int dependency = getDependencyInsideIndex(action, actionType, dependencyRelations.size());
                    if (!(oracleMap.containsKey(feat) && oracleMap.get(feat).equals(predictedMap.get(feat))))
                        if (isBinary) {
                            bClassifier.changeWeight(actionType, f, feature, dependency, -predictedMap.get(feat));
                        } else {
                            classifier.changeWeight(actionType, f, feature, dependency, -predictedMap.get(feat));
                        }
                }
            }
            for (Pair<Integer, Object> feat : oracleMap.keySet()) {
                if (feat.second != null) {
                    int action = feat.first;
                    Object feature = feat.second;
                    Actions actionType = Actions.intToAction(action, dependencyRelations.size());
                    int dependency = getDependencyInsideIndex(action, actionType, dependencyRelations.size());
                    if (!(predictedMap.containsKey(feat) && predictedMap.get(feat).equals(oracleMap.get(feat))))
                        if (isBinary) {
                            bClassifier.changeWeight(actionType, f, feature, dependency, oracleMap.get(feat));
                        } else {
                            classifier.changeWeight(actionType, f, feature, dependency, oracleMap.get(feat));
                        }
                }
            }
        }
    }

    private boolean isOracle(Configuration bestConfiguration) {
        /*int lastAction = bestConfiguration.actionHistory.get(bestConfiguration.actionHistory.size() - 1);
        Object[] features = FeatureExtractor.extractAllParseFeatures(bestConfiguration, featureLength);
        float score;
        if (lastAction == 0) {
            score = bClassifier.shiftScore(features, false);
        } else if (lastAction == 1) {
            score = bClassifier.reduceScore(features, false);
        } else if ((lastAction - 3 - label) == 0) {
            float[] rightArcScores = bClassifier.rightArcScores(features, false);
            score = rightArcScores[label];
        } else {
            float[] leftArcScores = bClassifier.leftArcScores(features, false);
            score = leftArcScores[label];
        }*/
        ArrayList<Integer> actions = bestConfiguration.actionHistory;
        Object[] features = FeatureExtractor.extractAllParseFeatures(bestConfiguration, featureLength);
        float score = 0f;
        int label;
        for (int action:actions) {
            if (action == 0) {
                score += bClassifier.shiftScore(features, true);
            } else if (action == 1) {
                score += bClassifier.reduceScore(features, true);
            } else if (action >= 3 + dependencyRelations.size()) {
                label = action - (3 + dependencyRelations.size());
                float[] leftArcScores = bClassifier.leftArcScores(features, true);
                score += leftArcScores[label];
            } else {
                label = action - 3;
                float[] rightArcScores = bClassifier.rightArcScores(features, true);
                score += rightArcScores[label];
            }
        }
        return (score >= 0);
    }
}