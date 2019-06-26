package YaraParser.TransitionBasedSystem.Configuration;

import YaraParser.Accessories.Pair;

import java.util.ArrayList;
import java.util.HashMap;

public class CompactTree {
    public HashMap<Integer, Pair<Integer, String>> goldDependencies;
    public ArrayList<String> posTags;

    public CompactTree(HashMap<Integer, Pair<Integer, String>> goldDependencies, ArrayList<String> posTags) {
        this.goldDependencies = goldDependencies;
        this.posTags = posTags;
    }
}
