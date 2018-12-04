package YaraParser.Structures;

import YaraParser.Accessories.Options;
import YaraParser.Learning.AveragedPerceptron;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Created by Mohammad Sadegh Rasooli. ML-NLP Lab, Department of Computer
 * Science, Columbia University Date Created: 1/8/15 Time: 11:41 AM To report
 * any bugs or problems contact rasooli@cs.columbia.edu
 */

public class InfStruct {
    public List<HashMap<Object, Float>> shiftFeatureAveragedWeights;
    public List<HashMap<Object, Float>> reduceFeatureAveragedWeights;
    public List<HashMap<Object, CompactArray>> leftArcFeatureAveragedWeights;
    public List<HashMap<Object, CompactArray>> rightArcFeatureAveragedWeights;
    public int dependencySize;

    public IndexMaps maps;
    public ArrayList<Integer> dependencyLabels;
    public Options options;

    public InfStruct(AveragedPerceptron perceptron, IndexMaps maps, ArrayList<Integer> dependencyLabels,
                     Options options) {
        shiftFeatureAveragedWeights = new ArrayList<>(perceptron.shiftFeatureAveragedWeights.size());
        reduceFeatureAveragedWeights = new ArrayList<>(perceptron.reduceFeatureAveragedWeights.size());

        this.dependencySize = perceptron.dependencySize;

        for (int i = 0; i < perceptron.shiftFeatureAveragedWeights.size(); i++) {
            shiftFeatureAveragedWeights.add(i, new HashMap<>());
            for (Object feat : perceptron.shiftFeatureWeights.get(i).keySet()) {
                float vals = perceptron.shiftFeatureWeights.get(i).get(feat);
                float avgVals = perceptron.shiftFeatureAveragedWeights.get(i).get(feat);
                float newVals = vals - (avgVals / perceptron.iteration);
                shiftFeatureAveragedWeights.get(i).put(feat, newVals);
            }
        }

        for (int i = 0; i < perceptron.reduceFeatureAveragedWeights.size(); i++) {
            reduceFeatureAveragedWeights.add(i, new HashMap<>());
            for (Object feat : perceptron.reduceFeatureWeights.get(i).keySet()) {
                float vals = perceptron.reduceFeatureWeights.get(i).get(feat);
                float avgVals = perceptron.reduceFeatureAveragedWeights.get(i).get(feat);
                float newVals = vals - (avgVals / perceptron.iteration);
                reduceFeatureAveragedWeights.get(i).put(feat, newVals);
            }
        }

        leftArcFeatureAveragedWeights = new ArrayList<>(perceptron.leftArcFeatureAveragedWeights.size());

        for (int i = 0; i < perceptron.leftArcFeatureAveragedWeights.size(); i++) {
            leftArcFeatureAveragedWeights.add(i, new HashMap<>());
            for (Object feat : perceptron.leftArcFeatureWeights.get(i).keySet()) {
                CompactArray vals = perceptron.leftArcFeatureWeights.get(i).get(feat);
                CompactArray avgVals = perceptron.leftArcFeatureAveragedWeights.get(i).get(feat);
                leftArcFeatureAveragedWeights.get(i).put(feat,
                        getAveragedCompactArray(vals, avgVals, perceptron.iteration));
            }
        }

        rightArcFeatureAveragedWeights = new ArrayList<>(perceptron.rightArcFeatureAveragedWeights.size());

        for (int i = 0; i < perceptron.rightArcFeatureAveragedWeights.size(); i++) {
            rightArcFeatureAveragedWeights.add(i, new HashMap<>());
            for (Object feat : perceptron.rightArcFeatureWeights.get(i).keySet()) {
                CompactArray vals = perceptron.rightArcFeatureWeights.get(i).get(feat);
                CompactArray avgVals = perceptron.rightArcFeatureAveragedWeights.get(i).get(feat);
                rightArcFeatureAveragedWeights.get(i).put(feat,
                        getAveragedCompactArray(vals, avgVals, perceptron.iteration));
            }
        }

        this.maps = maps;
        this.dependencyLabels = dependencyLabels;
        this.options = options;
    }

    public InfStruct(String modelPath) throws Exception {
        FileInputStream fos = new FileInputStream(modelPath);
        GZIPInputStream gz = new GZIPInputStream(fos);

        ObjectInputStream reader = new ObjectInputStream(gz);
        dependencyLabels = (ArrayList<Integer>) reader.readObject();
        maps = (IndexMaps) reader.readObject();
        options = (Options) reader.readObject();
        shiftFeatureAveragedWeights = (List<HashMap<Object, Float>>) reader.readObject();
        reduceFeatureAveragedWeights = (List<HashMap<Object, Float>>) reader.readObject();
        leftArcFeatureAveragedWeights = (List<HashMap<Object, CompactArray>>) reader.readObject();
        rightArcFeatureAveragedWeights = (List<HashMap<Object, CompactArray>>) reader.readObject();
        dependencySize = reader.readInt();
    }

    public void saveModel(String modelPath) throws Exception {
        FileOutputStream fos = new FileOutputStream(modelPath);
        GZIPOutputStream gz = new GZIPOutputStream(fos);

        ObjectOutput writer = new ObjectOutputStream(gz);
        writer.writeObject(dependencyLabels);
        writer.writeObject(maps);
        writer.writeObject(options);
        writer.writeObject(shiftFeatureAveragedWeights);
        writer.writeObject(reduceFeatureAveragedWeights);
        writer.writeObject(leftArcFeatureAveragedWeights);
        writer.writeObject(rightArcFeatureAveragedWeights);
        writer.writeInt(dependencySize);
        writer.close();
    }

    private CompactArray getAveragedCompactArray(CompactArray ca, CompactArray aca, int iteration) {
        int offset = ca.getOffset();
        float[] a = ca.getArray();
        float[] aa = aca.getArray();
        float[] aNew = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            aNew[i] = a[i] - (aa[i] / iteration);
        }
        return new CompactArray(offset, aNew);
    }
}
