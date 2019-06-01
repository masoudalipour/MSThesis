package YaraParser.TransitionBasedSystem.Parser;

import YaraParser.TransitionBasedSystem.Configuration.Configuration;

import java.util.ArrayList;

class GeneticAlg {
    private Configuration initConfiguration;
    private ArrayList<Integer> initGenome;

    GeneticAlg(Configuration config) {
        initConfiguration = config;
        initGenome = config.actionHistory;
    }

    public Configuration getConfiguration() {
        Configuration outputConfig = initConfiguration;

        return outputConfig;
    }
}
