package YaraParser.TransitionBasedSystem.Parser;

import YaraParser.Accessories.Pair;
import YaraParser.Learning.AveragedPerceptron;
import YaraParser.Learning.BinaryPerceptron;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.State;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;

import java.util.ArrayList;
import java.util.TreeSet;
import java.util.stream.IntStream;

public class GeneticAlg {
    private final boolean rootFirst;
    private final ArrayList<Integer> dependencyRelations;
    private ArrayList<Configuration> initConfigurations;
    private BinaryPerceptron binaryClassifier;
    private AveragedPerceptron yaraClassifier;

    public GeneticAlg(ArrayList<Configuration> configs, BinaryPerceptron binaryPerceptron,
                      AveragedPerceptron averagedPerceptron, final boolean rootFirst,
                      final ArrayList<Integer> dependencyRelations) {
        initConfigurations = configs;
        binaryClassifier = binaryPerceptron;
        yaraClassifier = averagedPerceptron;
        this.rootFirst = rootFirst;
        this.dependencyRelations = dependencyRelations;
    }

    public Configuration getConfiguration() throws Exception {
        ArrayList<Configuration> nextGen = initConfigurations;
        /*for (Configuration config : initConfigurations) {
            population.add(new GeneticElement(config.actionHistory, getActionsScore(config)));
        }*/
        for (int i = 0; i < 10; i++) {
            TreeSet<Configuration> population = new TreeSet<>();
            for (Configuration config : nextGen) {
                ArrayList<Float> scores = getActionsScore(config);
                Configuration c = mutate(config, findWorstAction(scores));
                ParseThread pt = new ParseThread(1, binaryClassifier, yaraClassifier, dependencyRelations,
                        binaryClassifier.featureSize(), c.sentence, rootFirst, 128);
                Pair<Configuration, Integer> configurationIntegerPair = pt.parse(c);
                population.add(configurationIntegerPair.first);
                if (population.size() > 128) {
                    population.pollFirst();
                }
            }
            nextGen = new ArrayList<>(population.descendingSet());
        }
        float bestScoreConf = Float.NEGATIVE_INFINITY;
        Configuration bestConf = nextGen.get(0);
        for (Configuration c : nextGen) {
            if (c.score > bestScoreConf) {
                bestConf = c;
                bestScoreConf = c.score;
            }
        }
        return bestConf;
    }

    private ArrayList<Float> getActionsScore(Configuration configuration) {
        ArrayList<Float> scores = new ArrayList<>(configuration.actionHistory.size());
        Configuration currentConfiguration = new Configuration(configuration.sentence, rootFirst);
        for (int action : configuration.actionHistory) {
            float score;
            State currentState = currentConfiguration.state;
            Object[] features = FeatureExtractor.extractAllParseFeatures(currentConfiguration,
                    binaryClassifier.featureSize());
            if (action == 0) {
                score = binaryClassifier.shiftScore(features, true);
                ArcEager.shift(currentState);
                currentConfiguration.addAction(action);
            } else if (action == 1) {
                score = binaryClassifier.reduceScore(features, true);
                ArcEager.reduce(currentState);
                currentConfiguration.addAction(action);
            } else if (action >= 3 + dependencyRelations.size()) {
                int label = action - (3 + dependencyRelations.size());
                float[] leftArcScores = binaryClassifier.leftArcScores(features, true);
                score = leftArcScores[label];
                ArcEager.leftArc(currentState, label);
                currentConfiguration.addAction(action);
            } else {
                int label = action - 3;
                float[] rightArcScores = binaryClassifier.rightArcScores(features, true);
                score = rightArcScores[label];
                ArcEager.rightArc(currentState, label);
                currentConfiguration.addAction(action);
            }
            scores.add(score);
        }
        return scores;
    }

    private Configuration mutate(Configuration configuration, int mutateIndex) {
        Configuration config = parse(configuration, new ArrayList<>(configuration.actionHistory.subList(0,
                mutateIndex - 1)));
        boolean canShift = ArcEager.canDo(Actions.Shift, config.state);
        boolean canReduce = ArcEager.canDo(Actions.Reduce, config.state);
        boolean canRightArc = ArcEager.canDo(Actions.RightArc, config.state);
        boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, config.state);
        ArrayList<Integer> actions = new ArrayList<>();
        ArrayList<Float> actionsScore = new ArrayList<>();
        float scoresSum = 0;
        Object[] features = FeatureExtractor.extractAllParseFeatures(config, binaryClassifier.featureSize());
        if (canShift && binaryClassifier.shiftScore(features, true) >= 0) {
            actions.add(0);
            actionsScore.add(yaraClassifier.shiftScore(features, true));
        }

        if (canReduce && binaryClassifier.reduceScore(features, true) > 0) {
            actions.add(1);
            actionsScore.add(yaraClassifier.reduceScore(features, true));
        }
        if (canRightArc) {
            float[] binaryScores = binaryClassifier.rightArcScores(features, true);
            float[] yaraScores = yaraClassifier.rightArcScores(features, true);
            for (int i = 0; i < binaryScores.length; i++) {
                if (binaryScores[i] > 0) {
                    actions.add(3 + i);
                    actionsScore.add(yaraScores[i]);
                }
            }
        }
        if (canLeftArc) {
            float[] binaryScores = binaryClassifier.leftArcScores(features, true);
            float[] yaraScores = yaraClassifier.leftArcScores(features, true);
            for (int i = 0; i < binaryScores.length; i++) {
                if (binaryScores[i] > 0) {
                    actions.add(3 + i);
                    actionsScore.add(yaraScores[i]);
                }
            }
        }
        // remove negative scores
        IntStream.range(0, actionsScore.size()).filter(i -> actionsScore.get(i) < 0).forEach(i -> {
            actionsScore.remove(i);
            actions.remove(i);
        });
        for (float actionScore : actionsScore) {
            scoresSum += actionScore;
        }
        actionsScore.set(0, actionsScore.get(0) / scoresSum);
        for (int i = 1; i < actionsScore.size(); i++) {
            float s = actionsScore.get(i) / scoresSum + actionsScore.get(i - 1);
            actionsScore.set(i, s);
        }
        float ind = (float) Math.random();
        for (int i = 0; i < actionsScore.size(); i++) {
            if (actionsScore.get(i) >= ind) {
                ArrayList<Integer> action = new ArrayList<>();
                action.add(actions.get(i));
                parse(config, action);
            }
        }
        return config;
    }

    /**
     * Parse a given sentence with a sequence of actions
     *
     * @param currentConfiguration the initial configuration
     * @param actionHistory        the sequence of actions that will apply on currentConfiguration
     * @return the parsed configuration
     */
    private Configuration parse(Configuration currentConfiguration, ArrayList<Integer> actionHistory) {
        for (int action : actionHistory) {
            State currentState = currentConfiguration.state;
            if (action == 0) {
                ArcEager.shift(currentState);
                currentConfiguration.addAction(action);
            } else if (action == 1) {
                ArcEager.reduce(currentState);
                currentConfiguration.addAction(action);
            } else if (action >= 3 + dependencyRelations.size()) {
                int label = action - (3 + dependencyRelations.size());
                ArcEager.leftArc(currentState, label);
                currentConfiguration.addAction(action);
            } else {
                int label = action - 3;
                ArcEager.rightArc(currentState, label);
                currentConfiguration.addAction(action);
            }
        }
        return currentConfiguration;
    }

    private int findWorstAction(ArrayList<Float> actions) {
        int worstAction = 0;
        float worstScore = Float.POSITIVE_INFINITY;
        for (int i = 0; i < actions.size(); i++) {
            if (actions.get(i) < worstScore) {
                worstAction = i;
                worstScore = actions.get(i);
            }
        }
        return worstAction;
    }
}
