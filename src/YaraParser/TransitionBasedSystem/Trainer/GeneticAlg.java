package YaraParser.TransitionBasedSystem.Trainer;

import YaraParser.Learning.AveragedPerceptron;
import YaraParser.Learning.BinaryPerceptron;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GeneticElement;
import YaraParser.TransitionBasedSystem.Configuration.State;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import YaraParser.TransitionBasedSystem.Parser.ArcEager;

import java.util.ArrayList;
import java.util.TreeSet;

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
        TreeSet<GeneticElement> population = new TreeSet<>();
        ArrayList<Configuration> nextGen = new ArrayList<>();
        for (Configuration config: initConfigurations) {
            population.add(new GeneticElement(config.actionHistory, getActionsStatus(config)));
        }

        return nextGen.get(0);
    }

    private ArrayList<Boolean> getActionsStatus(Configuration configuration) {
        ArrayList<Boolean> isRightAction = new ArrayList<>(configuration.actionHistory.size());
        Configuration currentConfiguration = new Configuration(configuration.sentence, rootFirst);
        for (int i = 0; i < configuration.actionHistory.size(); i++) {
            int action = configuration.actionHistory.get(i);
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
            if (score < 0) {
                isRightAction.set(i, false);
            } else {
                isRightAction.set(i, true);
            }
        }
        return isRightAction;
    }
}
