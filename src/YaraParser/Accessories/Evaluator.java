package YaraParser.Accessories;

import YaraParser.TransitionBasedSystem.Configuration.CompactTree;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Evaluator {
    public static void evaluate(String testPath, String predictedPath, HashSet<String> puncTags) throws Exception {
        CoNLLReader goldReader = new CoNLLReader(testPath);
        CoNLLReader predictedReader = new CoNLLReader(predictedPath);
        ArrayList<CompactTree> goldConfiguration = goldReader.readStringData();
        ArrayList<CompactTree> predConfiguration = predictedReader.readStringData();
        goldReader.close();
        predictedReader.close();
        double unlabeledMatch = 0;
        double labeledMatch = 0;
        int all = 0;
        double fullUnlabeledMatch = 0;
        double fullLabMatch = 0;
        int numTree = 0;
        BufferedWriter writer = new BufferedWriter(new FileWriter("evaluation.log", true));
        for (int i = 0; i < predConfiguration.size(); i++) {
            HashMap<Integer, Pair<Integer, String>> goldDeps = goldConfiguration.get(i).goldDependencies;
            HashMap<Integer, Pair<Integer, String>> predDeps = predConfiguration.get(i).goldDependencies;
            ArrayList<String> goldTags = goldConfiguration.get(i).posTags;
            numTree++;
            boolean isFullMatch = true;
            boolean isUnlabeledMatch = true;
            for (int dep : goldDeps.keySet()) {
                if (!puncTags.contains(goldTags.get(dep - 1).trim())) {
                    all++;
                    int gh = goldDeps.get(dep).first;
                    int ph = predDeps.get(dep).first;
                    String gl = goldDeps.get(dep).second.trim();
                    String pl = predDeps.get(dep).second.trim();
                    if (ph == gh) {
                        unlabeledMatch++;
                        if (pl.equals(gl)) {
                            labeledMatch++;
                            writer.write("sentence " + i + " parsed right");
                        } else {
                            isFullMatch = false;
                        }
                    } else {
                        isFullMatch = false;
                        isUnlabeledMatch = false;
                        writer.write("sentence " + i + " parsed wrong");
                    }
                }
            }
            if (isFullMatch) {
                fullLabMatch++;
            }
            if (isUnlabeledMatch) {
                fullUnlabeledMatch++;
            }
        }
        DecimalFormat decimalFormat = new DecimalFormat("0.00%");
        double labeledAccuracy = labeledMatch / all;
        double unlabeledAccuracy = unlabeledMatch / all;
        System.out.println("Labeled accuracy: " + decimalFormat.format(labeledAccuracy));
        System.out.println("Unlabeled accuracy:  " + decimalFormat.format(unlabeledAccuracy));
        double labExact = fullLabMatch / numTree;
        double unlabExact = fullUnlabeledMatch / numTree;
        System.out.println("Labeled exact match:  " + decimalFormat.format(labExact));
        System.out.println("Unlabeled exact match:  " + decimalFormat.format(unlabExact));
        System.out.println();
    }
}
