package YaraParser.TransitionBasedSystem.Configuration;

import java.util.ArrayList;

public class GeneticElement implements Comparable<GeneticElement> {
    ArrayList<Float> actionsScore;
    ArrayList<Integer> actions;
    float score;

    GeneticElement(ArrayList<Integer> actionHistory) {
        actions = actionHistory;
    }

    public GeneticElement(ArrayList<Integer> actionHistory, ArrayList<Float> scores) throws Exception {
        if (actionHistory.size() != scores.size()) {
            throw new Exception("The array of actions and the array of actions' score have different lengths");
        }
        actions = actionHistory;
        actionsScore = scores;
        for (float scr : actionsScore) {
            score += scr;
        }
    }

    @Override
    public int compareTo(GeneticElement element) {
        float diff = score - element.score;
        if (diff > 0) {
            return 2;
        } else if (diff < 0) {
            return -2;
        }
        return 0;
    }
}
