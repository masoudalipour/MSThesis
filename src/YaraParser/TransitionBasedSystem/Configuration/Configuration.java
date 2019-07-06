package YaraParser.TransitionBasedSystem.Configuration;

import YaraParser.Structures.Sentence;

import java.io.Serializable;
import java.util.ArrayList;

public class Configuration implements Comparable, Cloneable, Serializable {
    public Sentence sentence;
    public State state;
    /**
     * 0 = shift, 1 = reduce, 2 = unshift, 3 - dependencyLabels.size() = right arc and the rest of it is left arc
     */
    public ArrayList<Integer> actionHistory;
    public ArrayList<Integer> tabooList;
    public float score;

    public Configuration(Sentence sentence, boolean rootFirst) {
        this.sentence = sentence;
        state = new State(sentence.size(), rootFirst);
        score = (float) 0.0;
        actionHistory = new ArrayList<>(2 * (sentence.size() + 1));
        tabooList = new ArrayList<>();
    }

    public Configuration(Sentence sentence) {
        this.sentence = sentence;
        state = new State(sentence.size());
        score = (float) 0.0;
        actionHistory = new ArrayList<>(2 * (sentence.size() + 1));
        tabooList = new ArrayList<>();
    }

    /**
     * Returns the current score of the configuration
     *
     * @return float
     */
    public float getScore() {
        return score;
    }

    public void setScore(float score) {
        this.score = score;
    }

    public void addScore(float score) {
        this.score += score;
    }

    public void addAction(int action) {
        actionHistory.add(action);
    }

    @Override
    public int compareTo(Object o) {
        if (!(o instanceof Configuration)) {
            return hashCode() - o.hashCode();
        }
        // may be unsafe
        Configuration configuration = (Configuration) o;
        float diff = getScore() - configuration.getScore();
        if (diff > 0) {
            return (int) Math.ceil(diff);
        } else if (diff < 0) {
            return (int) Math.floor(diff);
        } else {
            return 0;
        }
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof Configuration) {
            Configuration configuration = (Configuration) o;
            if (configuration.score != score) {
                return false;
            }
            if (configuration.actionHistory.size() != actionHistory.size()) {
                return false;
            }
            for (int i = 0; i < actionHistory.size(); i++) {
                if (!actionHistory.get(i).equals(configuration.actionHistory.get(i))) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    @Override
    public Configuration clone() {
        Configuration configuration = new Configuration(sentence);
        ArrayList<Integer> history = new ArrayList<>(actionHistory.size());
        history.addAll(actionHistory);
        configuration.actionHistory = history;
        configuration.score = score;
        configuration.state = state.clone();
        return configuration;
    }

    @Override
    public int hashCode() {
        int hashCode = 0;
        int i = 0;
        for (int action : actionHistory) {
            hashCode += action << i++;
        }
        hashCode += score;
        return hashCode;
    }
}
