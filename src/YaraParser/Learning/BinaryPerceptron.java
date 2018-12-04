package YaraParser.Learning;

import YaraParser.Structures.CompactArray;
import YaraParser.Structures.InfStruct;
import YaraParser.TransitionBasedSystem.Parser.Actions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * This class tries to implement averaged Perceptron algorithm Collins, Michael.
 * "Discriminative training methods for hidden Markov models: Theory and
 * experiments with Perceptron algorithms." In Proceedings of the ACL-02
 * conference on Empirical methods in natural language processing-Volume 10, pp.
 * 1-8. Association for Computational Linguistics, 2002.
 * <p/>
 * The averaging update is also optimized by using the trick introduced in Hal
 * Daume's dissertation. For more information see the second chapter of his
 * thesis: Harold Charles Daume' III. "Practical Structured YaraParser.Learning
 * Techniques for Natural Language Processing", PhD thesis, ISI USC, 2006.
 * http://www.umiacs.umd.edu/~hal/docs/daume06thesis.pdf
 */
public class BinaryPerceptron {
    /**
     * For the weights for all features
     */
    public List<HashMap<Object, Float>> shiftFeatureWeights;
    public List<HashMap<Object, Float>> reduceFeatureWeights;
    public List<HashMap<Object, CompactArray>> leftArcFeatureWeights;
    public List<HashMap<Object, CompactArray>> rightArcFeatureWeights;

    public int iteration;
    public int dependencySize;
    /**
     * This is the main part of the extension to the original perceptron algorithm
     * which the averaging over all the history
     */
    public List<HashMap<Object, Float>> shiftFeatureAveragedWeights;
    public List<HashMap<Object, Float>> reduceFeatureAveragedWeights;
    public List<HashMap<Object, CompactArray>> leftArcFeatureAveragedWeights;
    public List<HashMap<Object, CompactArray>> rightArcFeatureAveragedWeights;

    public BinaryPerceptron(int featSize, int dependencySize) {
        shiftFeatureWeights = new ArrayList<>(featSize);
        reduceFeatureWeights = new ArrayList<>(featSize);
        leftArcFeatureWeights = new ArrayList<>(featSize);
        rightArcFeatureWeights = new ArrayList<>(featSize);

        shiftFeatureAveragedWeights = new ArrayList<>(featSize);
        reduceFeatureAveragedWeights = new ArrayList<>(featSize);
        leftArcFeatureAveragedWeights = new ArrayList<>(featSize);
        rightArcFeatureAveragedWeights = new ArrayList<>(featSize);
        for (int i = 0; i < featSize; i++) {
            shiftFeatureWeights.add(i, new HashMap<>());
            reduceFeatureWeights.add(i, new HashMap<>());
            leftArcFeatureWeights.add(i, new HashMap<>());
            rightArcFeatureWeights.add(i, new HashMap<>());

            shiftFeatureAveragedWeights.add(i, new HashMap<>());
            reduceFeatureAveragedWeights.add(i, new HashMap<>());
            leftArcFeatureAveragedWeights.add(i, new HashMap<>());
            rightArcFeatureAveragedWeights.add(i, new HashMap<>());
        }

        iteration = 1;
        this.dependencySize = dependencySize;
    }

    private BinaryPerceptron(List<HashMap<Object, Float>> shiftFeatureAveragedWeights,
                               List<HashMap<Object, Float>> reduceFeatureAveragedWeights,
                               List<HashMap<Object, CompactArray>> leftArcFeatureAveragedWeights,
                               List<HashMap<Object, CompactArray>> rightArcFeatureAveragedWeights, int dependencySize) {
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
        if (featureName == null)
            return;
        if (actionType == Actions.Shift) {
            if (!shiftFeatureWeights.get(slotNum).containsKey(featureName))
                shiftFeatureWeights.get(slotNum).put(featureName, change);
            else
                shiftFeatureWeights.get(slotNum).put(featureName,
                        shiftFeatureWeights.get(slotNum).get(featureName) + change);

            if (!shiftFeatureAveragedWeights.get(slotNum).containsKey(featureName))
                shiftFeatureAveragedWeights.get(slotNum).put(featureName, iteration * change);
            else
                shiftFeatureAveragedWeights.get(slotNum).put(featureName,
                        shiftFeatureAveragedWeights.get(slotNum).get(featureName) + iteration * change);
        } else if (actionType == Actions.Reduce) {
            if (!reduceFeatureWeights.get(slotNum).containsKey(featureName))
                reduceFeatureWeights.get(slotNum).put(featureName, change);
            else
                reduceFeatureWeights.get(slotNum).put(featureName, reduceFeatureWeights.get(slotNum).get(featureName) + change);

            if (!reduceFeatureAveragedWeights.get(slotNum).containsKey(featureName))
                reduceFeatureAveragedWeights.get(slotNum).put(featureName, iteration * change);
            else
                reduceFeatureAveragedWeights.get(slotNum).put(featureName,
                        reduceFeatureAveragedWeights.get(slotNum).get(featureName) + iteration * change);
        } else if (actionType == Actions.RightArc) {
            changeFeatureWeight(rightArcFeatureWeights.get(slotNum), rightArcFeatureAveragedWeights.get(slotNum), featureName,
                    labelIndex, change);
        } else if (actionType == Actions.LeftArc) {
            changeFeatureWeight(leftArcFeatureWeights.get(slotNum), leftArcFeatureAveragedWeights.get(slotNum), featureName,
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
            float[] val = new float[] { change };
            values = new CompactArray(labelIndex, val);
            map.put(featureName, values);

            float[] aVal = new float[] { iteration * change };
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

        List<HashMap<Object, Float>> map = decode ? shiftFeatureAveragedWeights : shiftFeatureWeights;

        for (int i = 0; i < features.length; i++) {
            if (features[i] == null || (i >= 26 && i < 32))
                continue;
            Float values = map.get(i).get(features[i]);
            if (values != null) {
                score += values;
            }
        }

        return score;
    }

    public float reduceScore(final Object[] features, boolean decode) {
        float score = 0.0f;

        List<HashMap<Object, Float>> map = decode ? reduceFeatureAveragedWeights : reduceFeatureWeights;

        for (int i = 0; i < features.length; i++) {
            if (features[i] == null || (i >= 26 && i < 32))
                continue;
            Float values = map.get(i).get(features[i]);
            if (values != null) {
                score += values;
            }
        }

        return score;
    }

    public float[] leftArcScores(final Object[] features, boolean decode) {
        float scores[] = new float[dependencySize];

        List<HashMap<Object, CompactArray>> map = decode ? leftArcFeatureAveragedWeights : leftArcFeatureWeights;

        for (int i = 0; i < features.length; i++) {
            if (features[i] == null)
                continue;
            CompactArray values = map.get(i).get(features[i]);
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
        float scores[] = new float[dependencySize];

        List<HashMap<Object, CompactArray>> map = decode ? rightArcFeatureAveragedWeights : rightArcFeatureWeights;

        for (int i = 0; i < features.length; i++) {
            if (features[i] == null)
                continue;
            CompactArray values = map.get(i).get(features[i]);
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
        return shiftFeatureAveragedWeights.size();
    }

    public int raSize() {
        int size = 0;
        for (int i = 0; i < leftArcFeatureAveragedWeights.size(); i++) {
            for (Object feat : rightArcFeatureAveragedWeights.get(i).keySet()) {
                size += rightArcFeatureAveragedWeights.get(i).get(feat).length();
            }
        }
        return size;
    }

    public int effectiveRaSize() {
        int size = 0;
        for (int i = 0; i < leftArcFeatureAveragedWeights.size(); i++) {
            for (Object feat : rightArcFeatureAveragedWeights.get(i).keySet()) {
                for (float f : rightArcFeatureAveragedWeights.get(i).get(feat).getArray())
                    if (f != 0f)
                        size++;
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
                for (float f : leftArcFeatureAveragedWeight.get(feat).getArray())
                    if (f != 0f)
                        size++;
            }
        }
        return size;
    }
}
