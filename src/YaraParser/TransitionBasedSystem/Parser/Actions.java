package YaraParser.TransitionBasedSystem.Parser;

/**
 * Created by Mohammad Sadegh Rasooli.
 * ML-NLP Lab, Department of Computer Science, Columbia University
 * Date Created: 12/23/14
 * Time: 11:08 AM
 * To report any bugs or problems contact rasooli@cs.columbia.edu
 */

public enum Actions {
    RightArc(2), LeftArc(3), Shift(0), Reduce(1), Unshift(4);

    private int value;

    Actions(int value) {
        this.value = value;
    }

    public static Actions intToAction(int action, int dependencyRelationsLength) {
        Actions actionType = Actions.Shift;
        if (action == 1) {
            actionType = Actions.Reduce;
        } else if (action >= 3 + dependencyRelationsLength) {
            actionType = Actions.LeftArc;
        } else if (action >= 3) {
            actionType = Actions.RightArc;
        } else if (action == 2) {
            actionType = Actions.Unshift;
        }
        return actionType;
    }
}
