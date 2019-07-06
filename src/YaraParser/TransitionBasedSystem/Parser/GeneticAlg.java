package YaraParser.TransitionBasedSystem.Parser;

import YaraParser.Accessories.Pair;
import YaraParser.Learning.AveragedPerceptron;
import YaraParser.Learning.BinaryPerceptron;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.State;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;

import java.util.ArrayList;
import java.util.TreeSet;

public class GeneticAlg {
    private final boolean rootFirst;
    private final ArrayList<Integer> dependencyRelations;
    private ArrayList<Configuration> initConfigurations;
    private BinaryPerceptron mammClassifier;
    private AveragedPerceptron yaraClassifier;
    private int generationSize;

    GeneticAlg(ArrayList<Configuration> configs, BinaryPerceptron binaryPerceptron,
               AveragedPerceptron averagedPerceptron, final boolean rootFirst,
               final ArrayList<Integer> dependencyRelations, final int genSize) {
        initConfigurations = configs;
        mammClassifier = binaryPerceptron;
        yaraClassifier = averagedPerceptron;
        this.rootFirst = rootFirst;
        this.dependencyRelations = dependencyRelations;
        generationSize = genSize;
    }

    public Configuration getConfiguration() {
        ArrayList<Configuration> nextGen = initConfigurations;
        /*for (Configuration config : initConfigurations) {
            population.add(new GeneticElement(config.actionHistory, getActionsScore(config)));
        }*/
        float highestScore = Float.NEGATIVE_INFINITY;
        int genWithoutEnhance = 0;
        while (genWithoutEnhance < 10) {
            TreeSet<Configuration> population = new TreeSet<>(nextGen);
            for (Configuration config : nextGen) {
                ArrayList<Float> scores = getActionsScore(config);
                int mutationIndex = findWorstAction(scores, config.tabooList);
                Configuration c = mutate(config, mutationIndex);
                // remove the indexes bigger than mutation point from the configuration's taboo list
                config.tabooList.removeIf(integer -> integer > mutationIndex);
                config.tabooList.add(mutationIndex);
                ParseThread pt = new ParseThread(1, mammClassifier,
                                                 yaraClassifier,
                                                 dependencyRelations,
                                                 mammClassifier.featureSize(),
                                                 c.sentence,
                                                 rootFirst,
                                                 8);
                Pair<Configuration, Integer> configurationIntegerPair = pt.parse(c);
                population.add(configurationIntegerPair.first);
                if (population.size() > generationSize) {
                    population.pollFirst();
                }
            }
            nextGen = new ArrayList<>(population.descendingSet());
            float thisGenBestScore = Float.NEGATIVE_INFINITY;
            for (Configuration c : nextGen) {
                if (c.score > thisGenBestScore) {
                    thisGenBestScore = c.score;
                }
            }
            if (highestScore == thisGenBestScore) {
                genWithoutEnhance++;
            } else {
                // log
                if (highestScore > Float.NEGATIVE_INFINITY) {
                    System.out.println("Improve with genetic by the amount of " + (thisGenBestScore - highestScore));
                }
                highestScore = thisGenBestScore;
                genWithoutEnhance = 0;
            }
        }
        highestScore = Float.NEGATIVE_INFINITY;
        Configuration bestConf = nextGen.get(0);
        for (Configuration c : nextGen) {
            if (c.score > highestScore) {
                bestConf = c;
                highestScore = c.score;
            }
        }
        return bestConf;
    }

    /**
     * receives a configuration and returns the MAMM model's score for each action in the configuration's action
     * history
     * @param configuration The configuration to be calculated it's action's sequence
     * @return The sequence of the action's score
     */
    private ArrayList<Float> getActionsScore(Configuration configuration) {
        ArrayList<Float> scores = new ArrayList<>(configuration.actionHistory.size());
        Configuration currentConfiguration = new Configuration(configuration.sentence, rootFirst);
        for (int action : configuration.actionHistory) {
            float score;
            State currentState = currentConfiguration.state;
            Object[] features = FeatureExtractor.extractAllParseFeatures(currentConfiguration,
                                                                         mammClassifier.featureSize());
            if (action == 0) {
                score = mammClassifier.shiftScore(features, true);
                ArcEager.shift(currentState);
                currentConfiguration.addAction(action);
            } else if (action == 1) {
                score = mammClassifier.reduceScore(features, true);
                ArcEager.reduce(currentState);
                currentConfiguration.addAction(action);
            } else if (action >= 3 + dependencyRelations.size()) {
                int label = action - (3 + dependencyRelations.size());
                float[] leftArcScores = mammClassifier.leftArcScores(features, true);
                score = leftArcScores[label];
                ArcEager.leftArc(currentState, label);
                currentConfiguration.addAction(action);
            } else {
                int label = action - 3;
                float[] rightArcScores = mammClassifier.rightArcScores(features, true);
                score = rightArcScores[label];
                ArcEager.rightArc(currentState, label);
                currentConfiguration.addAction(action);
            }
            scores.add(score);
        }
        return scores;
    }

    private Configuration mutate(Configuration configuration, int mutateIndex) {
        // log
        System.out.println("mutation start parse");
        Configuration config = parse(new Configuration(configuration.sentence, rootFirst),
                                     new ArrayList<>(configuration.actionHistory.subList(0, mutateIndex)));
        boolean canShift = ArcEager.canDo(Actions.Shift, config.state);
        boolean canReduce = ArcEager.canDo(Actions.Reduce, config.state);
        boolean canRightArc = ArcEager.canDo(Actions.RightArc, config.state);
        boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, config.state);
        ArrayList<Integer> actions = new ArrayList<>();
        ArrayList<Float> yaraActionsScore = new ArrayList<>();
        ArrayList<Float> mammActionsScore = new ArrayList<>();
        float scoresSum = 0;
        Object[] features = FeatureExtractor.extractAllParseFeatures(config, mammClassifier.featureSize());
        if (canShift) {
            actions.add(0);
            yaraActionsScore.add(yaraClassifier.shiftScore(features, true));
            mammActionsScore.add(mammClassifier.shiftScore(features, true));
        }

        if (canReduce) {
            actions.add(1);
            yaraActionsScore.add(yaraClassifier.reduceScore(features, true));
            mammActionsScore.add(mammClassifier.reduceScore(features, true));
        }
        if (canRightArc) {
            float[] mammScores = mammClassifier.rightArcScores(features, true);
            float[] yaraScores = yaraClassifier.rightArcScores(features, true);
            for (int dependency : dependencyRelations){
                actions.add(3+dependency);
                yaraActionsScore.add(yaraScores[dependency]);
                mammActionsScore.add(mammScores[dependency]);
            }
        }
        if (canLeftArc) {
            float[] mammScores = mammClassifier.leftArcScores(features, true);
            float[] yaraScores = yaraClassifier.leftArcScores(features, true);
            for (int dependency : dependencyRelations){
                actions.add(3+dependencyRelations.size()+dependency);
                yaraActionsScore.add(yaraScores[dependency]);
                mammActionsScore.add(mammScores[dependency]);
            }
        }
        // reduce negative scores
        for(int i=0; i<yaraActionsScore.size();i++){
            if(mammActionsScore.get(i) < 0){
                float reduceValue = yaraActionsScore.get(i);
                if(reduceValue > 0) {
                    reduceValue = reduceValue/3*2;
                    reduceValue = 0 - reduceValue;
                } else if(reduceValue<0) {
                    reduceValue = reduceValue/3*2;
                } else {
                    reduceValue = -50;
                }
                yaraActionsScore.set(i, yaraActionsScore.get(i) + reduceValue);
            }
        }
        /*IntStream.range(0, yaraActionsScore.size()).filter(i -> yaraActionsScore.get(i) < 0).forEach(i -> {
            yaraActionsScore.remove(i);
            actions.remove(i);
        });*/

        // roulette wheel
        for (float actionScore : yaraActionsScore) {
            scoresSum += actionScore;
        }
        yaraActionsScore.set(0, yaraActionsScore.get(0) / scoresSum);
        for (int i = 1; i < yaraActionsScore.size(); i++) {
            float s = yaraActionsScore.get(i) / scoresSum + yaraActionsScore.get(i - 1);
            yaraActionsScore.set(i, s);
        }
        float ind = (float) Math.random();
        for (int i = 0; i < yaraActionsScore.size(); i++) {
            if (yaraActionsScore.get(i) >= ind) {
                ArrayList<Integer> action = new ArrayList<>();
                action.add(actions.get(i));
                // log
                System.out.println("mutation end parse");
                parse(config, action);
                break;
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

    private int findWorstAction(ArrayList<Float> actions, ArrayList<Integer> tabuList) {
        int worstAction = -1;
        float worstScore = Float.POSITIVE_INFINITY;
        for (int i = 0; i < actions.size() && !tabuList.contains(i); i++) {
            if (actions.get(i) < worstScore) {
                worstAction = i;
                worstScore = actions.get(i);
            }
        }
        return worstAction;
    }
}
