package YaraParser.Learning;

import YaraParser.Structures.CompactArray;
import YaraParser.Structures.InfStruct;
import YaraParser.Structures.Sentence;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.State;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;
import YaraParser.TransitionBasedSystem.Parser.Actions;
import YaraParser.TransitionBasedSystem.Parser.ArcEager;

import java.util.ArrayList;
import java.util.HashMap;

public class BinaryPerceptron {
    public HashMap<Object, Float>[] shiftFeatureWeights;
    public HashMap<Object, Float>[] reduceFeatureWeights;
    public HashMap<Object, CompactArray>[] leftArcFeatureWeights;
    public HashMap<Object, CompactArray>[] rightArcFeatureWeights;

    public int iteration;
    public int dependencySize;
    /**
     * This is the main part of the extension to the original perceptron algorithm which the averaging over all the
     * history
     */
    public HashMap<Object, Float>[] shiftFeatureAveragedWeights;
    public HashMap<Object, Float>[] reduceFeatureAveragedWeights;
    public HashMap<Object, CompactArray>[] leftArcFeatureAveragedWeights;
    public HashMap<Object, CompactArray>[] rightArcFeatureAveragedWeights;

    public BinaryPerceptron(int featSize, int dependencySize) {
        shiftFeatureWeights = new HashMap[featSize];
        reduceFeatureWeights = new HashMap[featSize];
        leftArcFeatureWeights = new HashMap[featSize];
        rightArcFeatureWeights = new HashMap[featSize];
        shiftFeatureAveragedWeights = new HashMap[featSize];
        reduceFeatureAveragedWeights = new HashMap[featSize];
        leftArcFeatureAveragedWeights = new HashMap[featSize];
        rightArcFeatureAveragedWeights = new HashMap[featSize];
        for (int i = 0; i < featSize; i++) {
            shiftFeatureWeights[i] = new HashMap<>();
            reduceFeatureWeights[i] = new HashMap<>();
            leftArcFeatureWeights[i] = new HashMap<>();
            rightArcFeatureWeights[i] = new HashMap<>();
            shiftFeatureAveragedWeights[i] = new HashMap<>();
            reduceFeatureAveragedWeights[i] = new HashMap<>();
            leftArcFeatureAveragedWeights[i] = new HashMap<>();
            rightArcFeatureAveragedWeights[i] = new HashMap<>();
        }
        iteration = 1;
        this.dependencySize = dependencySize;
    }

    private BinaryPerceptron(HashMap<Object, Float>[] shiftFeatureAveragedWeights,
                             HashMap<Object, Float>[] reduceFeatureAveragedWeights,
                             HashMap<Object, CompactArray>[] leftArcFeatureAveragedWeights, HashMap<Object,
            CompactArray>[] rightArcFeatureAveragedWeights, int dependencySize) {
        this.shiftFeatureAveragedWeights = shiftFeatureAveragedWeights;
        this.reduceFeatureAveragedWeights = reduceFeatureAveragedWeights;
        this.leftArcFeatureAveragedWeights = leftArcFeatureAveragedWeights;
        this.rightArcFeatureAveragedWeights = rightArcFeatureAveragedWeights;
        this.dependencySize = dependencySize;
    }

    public BinaryPerceptron(InfStruct infStruct) {
        this(infStruct.shiftFeatureAveragedWeights, infStruct.reduceFeatureAveragedWeights,
                infStruct.leftArcFeatureAveragedWeights, infStruct.rightArcFeatureAveragedWeights,
                infStruct.dependencySize);
    }

    public void changeWeight(Actions actionType, int slotNum, Object featureName, int labelIndex, float change) {
        if (featureName == null) {
            return;
        }
        if (actionType == Actions.Shift) {
            if (!shiftFeatureWeights[slotNum].containsKey(featureName)) {
                shiftFeatureWeights[slotNum].put(featureName, change);
            } else {
                shiftFeatureWeights[slotNum].put(featureName, shiftFeatureWeights[slotNum].get(featureName) + change);
            }
            if (!shiftFeatureAveragedWeights[slotNum].containsKey(featureName)) {
                shiftFeatureAveragedWeights[slotNum].put(featureName, iteration * change);
            } else {
                shiftFeatureAveragedWeights[slotNum].put(featureName,
                        shiftFeatureAveragedWeights[slotNum].get(featureName) + iteration * change);
            }
        } else if (actionType == Actions.Reduce) {
            if (!reduceFeatureWeights[slotNum].containsKey(featureName)) {
                reduceFeatureWeights[slotNum].put(featureName, change);
            } else {
                reduceFeatureWeights[slotNum].put(featureName, reduceFeatureWeights[slotNum].get(featureName) + change);
            }
            if (!reduceFeatureAveragedWeights[slotNum].containsKey(featureName)) {
                reduceFeatureAveragedWeights[slotNum].put(featureName, iteration * change);
            } else {
                reduceFeatureAveragedWeights[slotNum].put(featureName,
                        reduceFeatureAveragedWeights[slotNum].get(featureName) + iteration * change);
            }
        } else if (actionType == Actions.RightArc) {
            changeFeatureWeight(rightArcFeatureWeights[slotNum], rightArcFeatureAveragedWeights[slotNum], featureName
                    , labelIndex, change);
        } else if (actionType == Actions.LeftArc) {
            changeFeatureWeight(leftArcFeatureWeights[slotNum], leftArcFeatureAveragedWeights[slotNum], featureName,
                    labelIndex, change);
        }
    }

    private void changeFeatureWeight(HashMap<Object, CompactArray> map, HashMap<Object, CompactArray> aMap,
                                     Object featureName, int labelIndex, float change) {
        CompactArray values = map.get(featureName);
        CompactArray aValues;
        if (values != null) {
            values.expandArray(labelIndex, change);
            aValues = aMap.get(featureName);
            aValues.expandArray(labelIndex, iteration * change);
        } else {
            float[] val = new float[]{change};
            values = new CompactArray(labelIndex, val);
            map.put(featureName, values);
            float[] aVal = new float[]{iteration * change};
            aValues = new CompactArray(labelIndex, aVal);
            aMap.put(featureName, aValues);
        }
    }

    /**
     * Adds to the iterations
     */
    public void incrementIteration() {
        iteration++;
    }

    public float shiftScore(final Object[] features, boolean decode) {
        float score = 0.0f;
        HashMap<Object, Float>[] map = decode ? shiftFeatureAveragedWeights : shiftFeatureWeights;
        for (int i = 0; i < features.length; i++) {
            if (features[i] == null || (i >= 26 && i < 32)) {
                continue;
            }
            Float values = map[i].get(features[i]);
            if (values != null) {
                score += values;
            }
        }
        return score;
    }

    public float reduceScore(final Object[] features, boolean decode) {
        float score = 0.0f;
        HashMap<Object, Float>[] map = decode ? reduceFeatureAveragedWeights : reduceFeatureWeights;
        for (int i = 0; i < features.length; i++) {
            if (features[i] == null || (i >= 26 && i < 32)) {
                continue;
            }
            Float values = map[i].get(features[i]);
            if (values != null) {
                score += values;
            }
        }
        return score;
    }

    public float[] leftArcScores(final Object[] features, boolean decode) {
        float[] scores = new float[dependencySize];
        HashMap<Object, CompactArray>[] map = decode ? leftArcFeatureAveragedWeights : leftArcFeatureWeights;
        for (int i = 0; i < features.length; i++) {
            if (features[i] == null) {
                continue;
            }
            CompactArray values = map[i].get(features[i]);
            if (values != null) {
                int offset = values.getOffset();
                float[] weightVector = values.getArray();
                for (int d = offset; d < offset + weightVector.length; d++) {
                    scores[d] += weightVector[d - offset];
                }
            }
        }
        return scores;
    }

    public float[] rightArcScores(final Object[] features, boolean decode) {
        float[] scores = new float[dependencySize];
        HashMap<Object, CompactArray>[] map = decode ? rightArcFeatureAveragedWeights : rightArcFeatureWeights;
        for (int i = 0; i < features.length; i++) {
            if (features[i] == null) {
                continue;
            }
            CompactArray values = map[i].get(features[i]);
            if (values != null) {
                int offset = values.getOffset();
                float[] weightVector = values.getArray();
                for (int d = offset; d < offset + weightVector.length; d++) {
                    scores[d] += weightVector[d - offset];
                }
            }
        }
        return scores;
    }

    public int featureSize() {
        return shiftFeatureAveragedWeights.length;
    }

    public int raSize() {
        int size = 0;
        for (int i = 0; i < leftArcFeatureAveragedWeights.length; i++) {
            for (Object feat : rightArcFeatureAveragedWeights[i].keySet()) {
                size += rightArcFeatureAveragedWeights[i].get(feat).length();
            }
        }
        return size;
    }

    public int effectiveRaSize() {
        int size = 0;
        for (int i = 0; i < leftArcFeatureAveragedWeights.length; i++) {
            for (Object feat : rightArcFeatureAveragedWeights[i].keySet()) {
                for (float f : rightArcFeatureAveragedWeights[i].get(feat).getArray()) {
                    if (f != 0f) {
                        size++;
                    }
                }
            }
        }
        return size;
    }

    public int laSize() {
        int size = 0;
        for (HashMap<Object, CompactArray> leftArcFeatureAveragedWeight : leftArcFeatureAveragedWeights) {
            for (Object feat : leftArcFeatureAveragedWeight.keySet()) {
                size += leftArcFeatureAveragedWeight.get(feat).length();
            }
        }
        return size;
    }

    public int effectiveLaSize() {
        int size = 0;
        for (HashMap<Object, CompactArray> leftArcFeatureAveragedWeight : leftArcFeatureAveragedWeights) {
            for (Object feat : leftArcFeatureAveragedWeight.keySet()) {
                for (float f : leftArcFeatureAveragedWeight.get(feat).getArray()) {
                    if (f != 0f) {
                        size++;
                    }
                }
            }
        }
        return size;
    }

    public float calcScore(final boolean decode, final Sentence sentence, final boolean rootFirst,
                           final ArrayList<Integer> actionHistory, final ArrayList<Integer> dependencyRelations) {
        float score = 0f;
        Configuration currentConfiguration = new Configuration(sentence, rootFirst);
        for (int action : actionHistory) {
            State currentState = currentConfiguration.state;
            Object[] features = FeatureExtractor.extractAllParseFeatures(currentConfiguration, featureSize());
            if (action == 0) {
                score += shiftScore(features, decode);
                ArcEager.shift(currentState);
                currentConfiguration.addAction(action);
            } else if (action == 1) {
                score += reduceScore(features, decode);
                ArcEager.reduce(currentState);
                currentConfiguration.addAction(action);
            } else if (action >= 3 + dependencyRelations.size()) {
                int label = action - (3 + dependencyRelations.size());
                float[] leftArcScores = leftArcScores(features, decode);
                score += leftArcScores[label];
                ArcEager.leftArc(currentState, label);
                currentConfiguration.addAction(action);
            } else {
                int label = action - 3;
                float[] rightArcScores = rightArcScores(features, decode);
                score += rightArcScores[label];
                ArcEager.rightArc(currentState, label);
                currentConfiguration.addAction(action);
            }
        }
        return score;
    }
}
