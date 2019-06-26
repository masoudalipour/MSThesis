package YaraParser.TransitionBasedSystem.Parser;

import YaraParser.Accessories.Pair;
import YaraParser.Learning.AveragedPerceptron;
import YaraParser.Learning.BinaryPerceptron;
import YaraParser.Structures.Sentence;
import YaraParser.TransitionBasedSystem.Configuration.BeamElement;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Configuration.State;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.TreeSet;
import java.util.concurrent.Callable;

public class ParseThread implements Callable<Pair<Configuration, Integer>> {
    private AveragedPerceptron classifier;
    private BinaryPerceptron bClassifier;
    private ArrayList<Integer> dependencyRelations;
    private int featureLength;
    private Sentence sentence;
    private boolean rootFirst;
    private int beamWidth;
    private GoldConfiguration goldConfiguration;
    private boolean partial;
    private int id;
    private String outputFile;

    ParseThread(int id, AveragedPerceptron classifier, ArrayList<Integer> dependencyRelations, int featureLength,
                Sentence sentence, boolean rootFirst, int beamWidth, GoldConfiguration goldConfiguration,
                boolean partial) {
        this.id = id;
        this.classifier = classifier;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        this.sentence = sentence;
        this.rootFirst = rootFirst;
        this.beamWidth = beamWidth;
        this.goldConfiguration = goldConfiguration;
        this.partial = partial;
    }

    ParseThread(int id, BinaryPerceptron bClassifier, AveragedPerceptron classifier,
                ArrayList<Integer> dependencyRelations, int featureLength, Sentence sentence, boolean rootFirst,
                int beamWidth, GoldConfiguration goldConfiguration, boolean partial, String outputFile) {
        this.id = id;
        this.classifier = classifier;
        this.bClassifier = bClassifier;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        this.sentence = sentence;
        this.rootFirst = rootFirst;
        this.beamWidth = beamWidth;
        this.goldConfiguration = goldConfiguration;
        this.partial = partial;
        this.outputFile = outputFile;
    }

    @Override
    public Pair<Configuration, Integer> call() throws Exception {
        if (!partial) {
            return parse();
        } else {
            return new Pair<>(parsePartial(), id);
        }
    }

    private Pair<Configuration, Integer> parse() throws Exception {
        Configuration initialConfiguration = new Configuration(sentence, rootFirst);
        ArrayList<Configuration> beam = new ArrayList<>(beamWidth);
        beam.add(initialConfiguration);
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile + ".parse.sentence" + id + ".log", true));
        int wrongParse;
        int totalWrongParse = 0;
        int rightParse;
        int beamCounter = 0;
        boolean topBeamIsOracle;
        while (ArcEager.isNotTerminal(beam)) {
            topBeamIsOracle = false;
            beamCounter++;
            if (beamWidth != 1) {
                TreeSet<BeamElement> beamPreserver = new TreeSet<>();
                for (int b = 0; b < beam.size(); b++) {
                    Configuration configuration = beam.get(b);
                    State currentState = configuration.state;
                    float prevScore = configuration.score;
                    boolean canShift = ArcEager.canDo(Actions.Shift, currentState);
                    boolean canReduce = ArcEager.canDo(Actions.Reduce, currentState);
                    boolean canRightArc = ArcEager.canDo(Actions.RightArc, currentState);
                    boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, currentState);
                    Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
                    if (!canShift && !canReduce && !canRightArc && !canLeftArc) {
                        beamPreserver.add(new BeamElement(prevScore, b, 4, -1));
                        /*if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();*/
                    }
                    if (canShift) {
                        float score = classifier.shiftScore(features, true);
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 0, -1));
                        /*if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();*/
                    }
                    if (canReduce) {
                        float score = classifier.reduceScore(features, true);
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 1, -1));
                        /*if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();*/
                    }
                    if (canRightArc) {
                        float[] rightArcScores = classifier.rightArcScores(features, true);
                        for (int dependency : dependencyRelations) {
                            float score = rightArcScores[dependency];
                            float addedScore = score + prevScore;
                            beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));
                            /*if (beamPreserver.size() > beamWidth)
                                beamPreserver.pollFirst();*/
                        }
                    }
                    if (canLeftArc) {
                        float[] leftArcScores = classifier.leftArcScores(features, true);
                        for (int dependency : dependencyRelations) {
                            float score = leftArcScores[dependency];
                            float addedScore = score + prevScore;
                            beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));
                            /*if (beamPreserver.size() > beamWidth)
                                beamPreserver.pollFirst();*/
                        }
                    }
                }
                ArrayList<Configuration> repBeam = new ArrayList<>(beamWidth);
                rightParse = 0;
                wrongParse = 0;
                float bestScore = beamPreserver.last().score;
                Configuration bestConfiguration = null;
                for (BeamElement beamElement : beamPreserver.descendingSet()) {
                    if (repBeam.size() >= beamWidth) {
                        break;
                    }
                    int b = beamElement.number;
                    int action = beamElement.action;
                    int label = beamElement.label;
                    float score = beamElement.score;
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
                    newConfig.setScore(score);
                    if (score == bestScore) {
                        bestConfiguration = newConfig;
                    }
                    if (isOracle(newConfig)) {
                        rightParse++;
                        repBeam.add(newConfig);
                    } else {
                        wrongParse++;
                        writer.write("Wrong configuration " + (wrongParse + rightParse) + " in beam " + beamCounter);
                        writer.newLine();
                    }
                }
                totalWrongParse += wrongParse;
                writer.write(wrongParse + "configuration was wrong in beam " + beamCounter);
                writer.newLine();
                beam = repBeam;
                if (isOracle(bestConfiguration)) {
                    topBeamIsOracle = true;
                }
                writer.write("in beam " + beamCounter + " top beam was oracle?" + topBeamIsOracle);
                writer.newLine();
            } else {
                Configuration configuration = beam.get(0);
                State currentState = configuration.state;
                Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
                float bestScore = Float.NEGATIVE_INFINITY;
                int bestAction = -1;
                boolean canShift = ArcEager.canDo(Actions.Shift, currentState);
                boolean canReduce = ArcEager.canDo(Actions.Reduce, currentState);
                boolean canRightArc = ArcEager.canDo(Actions.RightArc, currentState);
                boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, currentState);
                if (!canShift && !canReduce && !canRightArc && !canLeftArc) {
                    if (!currentState.stackEmpty()) {
                        ArcEager.unShift(currentState);
                        configuration.addAction(2);
                    } else if (!currentState.bufferEmpty() && currentState.stackEmpty()) {
                        ArcEager.shift(currentState);
                        configuration.addAction(0);
                    }
                }
                if (canShift) {
                    float score = classifier.shiftScore(features, true);
                    if (score > bestScore) {
                        bestScore = score;
                        bestAction = 0;
                    }
                }
                if (canReduce) {
                    float score = classifier.reduceScore(features, true);
                    if (score > bestScore) {
                        bestScore = score;
                        bestAction = 1;
                    }
                }
                if (canRightArc) {
                    float[] rightArcScores = classifier.rightArcScores(features, true);
                    for (int dependency : dependencyRelations) {
                        float score = rightArcScores[dependency];
                        if (score > bestScore) {
                            bestScore = score;
                            bestAction = 3 + dependency;
                        }
                    }
                }
                if (ArcEager.canDo(Actions.LeftArc, currentState)) {
                    float[] leftArcScores = classifier.leftArcScores(features, true);
                    for (int dependency : dependencyRelations) {
                        float score = leftArcScores[dependency];
                        if (score > bestScore) {
                            bestScore = score;
                            bestAction = 3 + dependencyRelations.size() + dependency;
                        }
                    }
                }
                if (bestAction != -1) {
                    int label;
                    if (bestAction == 0) {
                        ArcEager.shift(configuration.state);
                    } else if (bestAction == (1)) {
                        ArcEager.reduce(configuration.state);
                    } else if (bestAction >= 3 + dependencyRelations.size()) {
                        label = bestAction - (3 + dependencyRelations.size());
                        ArcEager.leftArc(configuration.state, label);
                    } else {
                        label = bestAction - 3;
                        ArcEager.rightArc(configuration.state, label);
                    }
                    configuration.addScore(bestScore);
                    configuration.addAction(bestAction);
                }
            }
        }
        /*Configuration bestConfiguration = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (Configuration configuration : beam) {
            if (configuration.getScore() > bestScore) {
                bestScore = configuration.getScore();
                bestConfiguration = configuration;
            }
        }*/
        writer.write("total of " + totalWrongParse + " configuration was wrong in parse process");
        writer.newLine();
        writer.write("the oracle predicts that ");
        writer.newLine();
        writer.close();
        return new Pair<>(beam.get(0), id);
    }

    private Configuration parsePartial() {
        Configuration initialConfiguration = new Configuration(sentence, rootFirst);
        boolean isNonProjective = false;
        if (goldConfiguration.isNonprojective()) {
            isNonProjective = true;
        }
        ArrayList<Configuration> beam = new ArrayList<>(beamWidth);
        beam.add(initialConfiguration);
        while (ArcEager.isNotTerminal(beam)) {
            TreeSet<BeamElement> beamPreserver = new TreeSet<>();
            parsePartialWithOneThread(beam, beamPreserver, isNonProjective, goldConfiguration, beamWidth);
            ArrayList<Configuration> repBeam = new ArrayList<>(beamWidth);
            for (BeamElement beamElement : beamPreserver.descendingSet()) {
                if (repBeam.size() >= beamWidth) {
                    break;
                }
                int b = beamElement.number;
                int action = beamElement.action;
                int label = beamElement.label;
                float score = beamElement.score;
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
                newConfig.setScore(score);
                repBeam.add(newConfig);
            }
            beam = repBeam;
        }
        Configuration bestConfiguration = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (Configuration configuration : beam) {
            if (configuration.getScore() > bestScore) {
                bestScore = configuration.getScore();
                bestConfiguration = configuration;
            }
        }
        return bestConfiguration;
    }

    private void parsePartialWithOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver,
                                           Boolean isNonProjective, GoldConfiguration goldConfiguration,
                                           int beamWidth) {
        for (int b = 0; b < beam.size(); b++) {
            Configuration configuration = beam.get(b);
            State currentState = configuration.state;
            float prevScore = configuration.score;
            boolean canShift = ArcEager.canDo(Actions.Shift, currentState);
            boolean canReduce = ArcEager.canDo(Actions.Reduce, currentState);
            boolean canRightArc = ArcEager.canDo(Actions.RightArc, currentState);
            boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, currentState);
            Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
            if (!canShift && !canReduce && !canRightArc && !canLeftArc) {
                beamPreserver.add(new BeamElement(prevScore, b, 4, -1));
                if (beamPreserver.size() > beamWidth) {
                    beamPreserver.pollFirst();
                }
            }
            if (canShift) {
                if (isNonProjective || goldConfiguration.actionCost(Actions.Shift, -1, currentState) == 0) {
                    float score = classifier.shiftScore(features, true);
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 0, -1));
                    if (beamPreserver.size() > beamWidth) {
                        beamPreserver.pollFirst();
                    }
                }
            }
            if (canReduce) {
                if (isNonProjective || goldConfiguration.actionCost(Actions.Reduce, -1, currentState) == 0) {
                    float score = classifier.reduceScore(features, true);
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 1, -1));
                    if (beamPreserver.size() > beamWidth) {
                        beamPreserver.pollFirst();
                    }
                }
            }
            if (canRightArc) {
                float[] rightArcScores = classifier.rightArcScores(features, true);
                for (int dependency : dependencyRelations) {
                    if (isNonProjective || goldConfiguration.actionCost(Actions.RightArc, dependency, currentState) == 0) {
                        float score = rightArcScores[dependency];
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));
                        if (beamPreserver.size() > beamWidth) {
                            beamPreserver.pollFirst();
                        }
                    }
                }
            }
            if (canLeftArc) {
                float[] leftArcScores = classifier.leftArcScores(features, true);
                for (int dependency : dependencyRelations) {
                    if (isNonProjective || goldConfiguration.actionCost(Actions.LeftArc, dependency, currentState) == 0) {
                        float score = leftArcScores[dependency];
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));
                        if (beamPreserver.size() > beamWidth) {
                            beamPreserver.pollFirst();
                        }
                    }
                }
            }
        }
        if (beamPreserver.size() == 0) {
            for (int b = 0; b < beam.size(); b++) {
                Configuration configuration = beam.get(b);
                State currentState = configuration.state;
                float prevScore = configuration.score;
                boolean canShift = ArcEager.canDo(Actions.Shift, currentState);
                boolean canReduce = ArcEager.canDo(Actions.Reduce, currentState);
                boolean canRightArc = ArcEager.canDo(Actions.RightArc, currentState);
                boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, currentState);
                Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
                if (!canShift && !canReduce && !canRightArc && !canLeftArc) {
                    beamPreserver.add(new BeamElement(prevScore, b, 4, -1));
                    if (beamPreserver.size() > beamWidth) {
                        beamPreserver.pollFirst();
                    }
                }
                if (canShift) {
                    float score = classifier.shiftScore(features, true);
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 0, -1));
                    if (beamPreserver.size() > beamWidth) {
                        beamPreserver.pollFirst();
                    }
                }
                if (canReduce) {
                    float score = classifier.reduceScore(features, true);
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 1, -1));
                    if (beamPreserver.size() > beamWidth) {
                        beamPreserver.pollFirst();
                    }
                }
                if (canRightArc) {
                    float[] rightArcScores = classifier.rightArcScores(features, true);
                    for (int dependency : dependencyRelations) {
                        float score = rightArcScores[dependency];
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));
                        if (beamPreserver.size() > beamWidth) {
                            beamPreserver.pollFirst();
                        }
                    }
                }
                if (canLeftArc) {
                    float[] leftArcScores = classifier.leftArcScores(features, true);
                    for (int dependency : dependencyRelations) {
                        float score = leftArcScores[dependency];
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));
                        if (beamPreserver.size() > beamWidth) {
                            beamPreserver.pollFirst();
                        }
                    }
                }
            }
        }
    }

    private boolean isOracle(Configuration configuration) throws Exception {
        if (configuration == null) {
            throw new Exception("The input of isOracle is null");
        }
        return bClassifier.calcScore(true, configuration.sentence, rootFirst, configuration.actionHistory,
                featureLength, dependencyRelations) >= 0;


        // int lastAction = configuration.actionHistory.get(configuration.actionHistory.size() - 1);
        // Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
        // float score;
        // int label;
        // if (lastAction == 0) {
        //     score = bClassifier.shiftScore(features, true);
        // } else if (lastAction == 1) {
        //     score = bClassifier.reduceScore(features, true);
        // } else if (lastAction >= 3 + dependencyRelations.size()) {
        //     label = lastAction - (3 + dependencyRelations.size());
        //     float[] leftArcScores = bClassifier.leftArcScores(features, true);
        //     score = leftArcScores[label];
        // } else {
        //     label = lastAction - 3;
        //     float[] rightArcScores = bClassifier.rightArcScores(features, true);
        //     score = rightArcScores[label];
        // }



        /*ArrayList<Integer> actions = configuration.actionHistory;
        Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
        float score = 0f;
        int label;
        for (int action : actions) {
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
        return (score >= 0);*/
    }
}
