package YaraParser.TransitionBasedSystem.Configuration;

import java.util.ArrayList;

public class GeneticElement implements Comparable<GeneticElement> {
    ArrayList<Boolean> isRightAction;
    ArrayList<Integer> actions;
    float score;

    GeneticElement(ArrayList<Integer> actionHistory) {
        actions = actionHistory;
    }

    public GeneticElement(ArrayList<Integer> actionHistory, ArrayList<Boolean> actionsStatus) throws Exception {
        if (actionHistory.size() != actionsStatus.size()) {
            throw new Exception("Actions' length and status' length are different");
        }
        actions = actionHistory;
        isRightAction = actionsStatus;
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
