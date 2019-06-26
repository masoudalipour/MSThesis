package YaraParser.TransitionBasedSystem.Parser;

import YaraParser.Accessories.CoNLLReader;
import YaraParser.Accessories.Pair;
import YaraParser.Learning.AveragedPerceptron;
import YaraParser.Learning.BinaryPerceptron;
import YaraParser.Structures.IndexMaps;
import YaraParser.Structures.InfStruct;
import YaraParser.Structures.Sentence;
import YaraParser.TransitionBasedSystem.Configuration.BeamElement;
import YaraParser.TransitionBasedSystem.Configuration.Configuration;
import YaraParser.TransitionBasedSystem.Configuration.GoldConfiguration;
import YaraParser.TransitionBasedSystem.Configuration.State;
import YaraParser.TransitionBasedSystem.Features.FeatureExtractor;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.TreeSet;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class KBeamArcEagerParser extends TransitionBasedParser {
    /**
     * Any kind of classifier that can give us scores
     */
    private AveragedPerceptron classifier;
    private BinaryPerceptron bClassifier;
    private ArrayList<Integer> dependencyRelations;
    private int featureLength;
    private IndexMaps maps;
    private ExecutorService executor;
    private CompletionService<ArrayList<BeamElement>> pool;

    public KBeamArcEagerParser(AveragedPerceptron classifier, ArrayList<Integer> dependencyRelations,
                               int featureLength, IndexMaps maps, int numOfThreads) {
        this.classifier = classifier;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        this.maps = maps;
        executor = Executors.newFixedThreadPool(numOfThreads);
        pool = new ExecutorCompletionService<>(executor);
    }

    public KBeamArcEagerParser(BinaryPerceptron bClassifier, AveragedPerceptron classifier, ArrayList<Integer> dependencyRelations,
                               int featureLength, IndexMaps maps, int numOfThreads) {
        this.classifier = classifier;
        this.bClassifier = bClassifier;
        this.dependencyRelations = dependencyRelations;
        this.featureLength = featureLength;
        this.maps = maps;
        executor = Executors.newFixedThreadPool(numOfThreads);
        pool = new ExecutorCompletionService<>(executor);
    }

    public static KBeamArcEagerParser createParser(String modelPath, int numOfThreads) throws Exception {
        InfStruct infStruct = new InfStruct(modelPath);
        ArrayList<Integer> dependencyLabels = infStruct.dependencyLabels;
        IndexMaps maps = infStruct.maps;
        AveragedPerceptron averagedPerceptron = new AveragedPerceptron(infStruct);
        int featureSize = averagedPerceptron.featureSize();
        return new KBeamArcEagerParser(averagedPerceptron, dependencyLabels, featureSize, maps, numOfThreads);
    }

    private void parseWithOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver,
                                    Sentence sentence, boolean rootFirst, int beamWidth) {
        for (int b = 0; b < beam.size(); b++) {
            Configuration configuration = beam.get(b);
            State currentState = configuration.state;
            float prevScore = configuration.score;
            boolean canShift = ArcEager.canDo(Actions.Shift, currentState);
            boolean canReduce = ArcEager.canDo(Actions.Reduce, currentState);
            boolean canRightArc = ArcEager.canDo(Actions.RightArc, currentState);
            boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, currentState);
            Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
            if (!canShift
                    && !canReduce
                    && !canRightArc
                    && !canLeftArc) {
                beamPreserver.add(new BeamElement(prevScore, b, 4, -1));
                if (beamPreserver.size() > beamWidth)
                    beamPreserver.pollFirst();
            }
            if (canShift) {
                float score = classifier.shiftScore(features, true);
                float addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 0, -1));
                if (beamPreserver.size() > beamWidth)
                    beamPreserver.pollFirst();
            }
            if (canReduce) {
                float score = classifier.reduceScore(features, true);
                float addedScore = score + prevScore;
                beamPreserver.add(new BeamElement(addedScore, b, 1, -1));
                if (beamPreserver.size() > beamWidth)
                    beamPreserver.pollFirst();
            }
            if (canRightArc) {
                float[] rightArcScores = classifier.rightArcScores(features, true);
                for (int dependency : dependencyRelations) {
                    float score = rightArcScores[dependency];
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));
                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }
            if (canLeftArc) {
                float[] leftArcScores = classifier.leftArcScores(features, true);
                for (int dependency : dependencyRelations) {
                    float score = leftArcScores[dependency];
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));
                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }
        }
    }

    public Configuration parse(Sentence sentence, boolean rootFirst, int beamWidth, int numOfThreads) throws Exception {
        Configuration initialConfiguration = new Configuration(sentence, rootFirst);
        ArrayList<Configuration> beam = new ArrayList<>(beamWidth);
        beam.add(initialConfiguration);
        while (ArcEager.isNotTerminal(beam)) {
            TreeSet<BeamElement> beamPreserver = new TreeSet<>();
            if (numOfThreads == 1) {
                parseWithOneThread(beam, beamPreserver, sentence, rootFirst, beamWidth);
            } else {
                for (int b = 0; b < beam.size(); b++) {
                    pool.submit(new BeamScorerThread(true, classifier, beam.get(b),
                            dependencyRelations, featureLength, b));
                }
                for (int b = 0; b < beam.size(); b++) {
                    for (BeamElement element : pool.take().get()) {
                        beamPreserver.add(element);
                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }
            ArrayList<Configuration> repBeam = new ArrayList<>(beamWidth);
            for (BeamElement beamElement : beamPreserver.descendingSet()) {
                if (repBeam.size() >= beamWidth)
                    break;
                int b = beamElement.number;
                int action = beamElement.action;
                int label = beamElement.label;
                float score = beamElement.score;
                Configuration newConfig = beam.get(b).clone();
                if (action == 0) {
                    ArcEager.shift(newConfig.state);
                    newConfig.addAction(0);
                } else if (action == 1) {
                    ArcEager.reduce(newConfig.state);
                    newConfig.addAction(1);
                } else if (action == 2) {
                    ArcEager.rightArc(newConfig.state, label);
                    newConfig.addAction(3 + label);
                } else if (action == 3) {
                    ArcEager.leftArc(newConfig.state, label);
                    newConfig.addAction(3 + dependencyRelations.size() + label);
                } else if (action == 4) {
                    ArcEager.unShift(newConfig.state);
                    newConfig.addAction(2);
                }
                newConfig.setScore(score);
                repBeam.add(newConfig);
            }
            beam = repBeam;
        }
        Configuration bestConfiguration = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (Configuration configuration : beam) {
            if (configuration.getScore() > bestScore) {
                bestScore = configuration.getScore();
                bestConfiguration = configuration;
            }
        }
        return bestConfiguration;
    }

    private void parsePartialWithOneThread(ArrayList<Configuration> beam, TreeSet<BeamElement> beamPreserver,
                                           Boolean isNonProjective, GoldConfiguration goldConfiguration,
                                           int beamWidth, boolean rootFirst) {
        for (int b = 0; b < beam.size(); b++) {
            Configuration configuration = beam.get(b);
            State currentState = configuration.state;
            float prevScore = configuration.score;
            boolean canShift = ArcEager.canDo(Actions.Shift, currentState);
            boolean canReduce = ArcEager.canDo(Actions.Reduce, currentState);
            boolean canRightArc = ArcEager.canDo(Actions.RightArc, currentState);
            boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, currentState);
            Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
            if (!canShift
                    && !canReduce
                    && !canRightArc
                    && !canLeftArc && rootFirst) {
                beamPreserver.add(new BeamElement(prevScore, b, 4, -1));
                if (beamPreserver.size() > beamWidth)
                    beamPreserver.pollFirst();
            }
            if (canShift) {
                if (isNonProjective || goldConfiguration.actionCost(Actions.Shift, -1, currentState) == 0) {
                    float score = classifier.shiftScore(features, true);
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 0, -1));
                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }
            if (canReduce) {
                if (isNonProjective || goldConfiguration.actionCost(Actions.Reduce, -1, currentState) == 0) {
                    float score = classifier.reduceScore(features, true);
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 1, -1));
                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
            }
            if (canRightArc) {
                float[] rightArcScores = classifier.rightArcScores(features, true);
                for (int dependency : dependencyRelations) {
                    if (isNonProjective || goldConfiguration.actionCost(Actions.RightArc, dependency, currentState) == 0) {
                        float score = rightArcScores[dependency];
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));
                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }
            if (canLeftArc) {
                float[] leftArcScores = classifier.leftArcScores(features, true);
                for (int dependency : dependencyRelations) {
                    if (isNonProjective || goldConfiguration.actionCost(Actions.LeftArc, dependency, currentState) == 0) {
                        float score = leftArcScores[dependency];
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));
                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }
        }
        if (beamPreserver.size() == 0) {
            for (int b = 0; b < beam.size(); b++) {
                Configuration configuration = beam.get(b);
                State currentState = configuration.state;
                float prevScore = configuration.score;
                boolean canShift = ArcEager.canDo(Actions.Shift, currentState);
                boolean canReduce = ArcEager.canDo(Actions.Reduce, currentState);
                boolean canRightArc = ArcEager.canDo(Actions.RightArc, currentState);
                boolean canLeftArc = ArcEager.canDo(Actions.LeftArc, currentState);
                Object[] features = FeatureExtractor.extractAllParseFeatures(configuration, featureLength);
                if (!canShift
                        && !canReduce
                        && !canRightArc
                        && !canLeftArc) {
                    beamPreserver.add(new BeamElement(prevScore, b, 4, -1));
                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
                if (canShift) {
                    float score = classifier.shiftScore(features, true);
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 0, -1));
                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
                if (canReduce) {
                    float score = classifier.reduceScore(features, true);
                    float addedScore = score + prevScore;
                    beamPreserver.add(new BeamElement(addedScore, b, 1, -1));
                    if (beamPreserver.size() > beamWidth)
                        beamPreserver.pollFirst();
                }
                if (canRightArc) {
                    float[] rightArcScores = classifier.rightArcScores(features, true);
                    for (int dependency : dependencyRelations) {
                        float score = rightArcScores[dependency];
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 2, dependency));
                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
                if (canLeftArc) {
                    float[] leftArcScores = classifier.leftArcScores(features, true);
                    for (int dependency : dependencyRelations) {
                        float score = leftArcScores[dependency];
                        float addedScore = score + prevScore;
                        beamPreserver.add(new BeamElement(addedScore, b, 3, dependency));
                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }
        }
    }

    private Configuration parsePartial(GoldConfiguration goldConfiguration, Sentence sentence, boolean rootFirst,
                                       int beamWidth, int numOfThreads) throws Exception {
        Configuration initialConfiguration = new Configuration(sentence, rootFirst);
        boolean isNonProjective = false;
        if (goldConfiguration.isNonprojective()) {
            isNonProjective = true;
        }
        ArrayList<Configuration> beam = new ArrayList<>(beamWidth);
        beam.add(initialConfiguration);
        while (ArcEager.isNotTerminal(beam)) {
            TreeSet<BeamElement> beamPreserver = new TreeSet<>();
            if (numOfThreads == 1) {
                parsePartialWithOneThread(beam, beamPreserver, isNonProjective, goldConfiguration, beamWidth,
                        rootFirst);
            } else {
                for (int b = 0; b < beam.size(); b++) {
                    pool.submit(new PartialTreeBeamScorerThread(true, classifier, goldConfiguration, beam.get(b),
                            dependencyRelations, featureLength, b));
                }
                for (int b = 0; b < beam.size(); b++) {
                    for (BeamElement element : pool.take().get()) {
                        beamPreserver.add(element);
                        if (beamPreserver.size() > beamWidth)
                            beamPreserver.pollFirst();
                    }
                }
            }
            ArrayList<Configuration> repBeam = new ArrayList<>(beamWidth);
            for (BeamElement beamElement : beamPreserver.descendingSet()) {
                if (repBeam.size() >= beamWidth)
                    break;
                int b = beamElement.number;
                int action = beamElement.action;
                int label = beamElement.label;
                float score = beamElement.score;
                Configuration newConfig = beam.get(b).clone();
                if (action == 0) {
                    ArcEager.shift(newConfig.state);
                    newConfig.addAction(0);
                } else if (action == 1) {
                    ArcEager.reduce(newConfig.state);
                    newConfig.addAction(1);
                } else if (action == 2) {
                    ArcEager.rightArc(newConfig.state, label);
                    newConfig.addAction(3 + label);
                } else if (action == 3) {
                    ArcEager.leftArc(newConfig.state, label);
                    newConfig.addAction(3 + dependencyRelations.size() + label);
                } else if (action == 4) {
                    ArcEager.unShift(newConfig.state);
                    newConfig.addAction(2);
                }
                newConfig.setScore(score);
                repBeam.add(newConfig);
            }
            beam = repBeam;
        }
        Configuration bestConfiguration = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (Configuration configuration : beam) {
            if (configuration.getScore() > bestScore) {
                bestScore = configuration.getScore();
                bestConfiguration = configuration;
            }
        }
        return bestConfiguration;
    }

    /**
     * Needs CoNLL 2006 format
     */
    public void parseCoNLLFile(String inputFile, String outputFile, boolean rootFirst, int beamWidth, boolean labeled
            , boolean lowerCased, int numThreads, boolean partial, String scorePath) throws Exception {
        if (numThreads == 1)
            parseCoNLLFileNoParallel(inputFile, outputFile, rootFirst, beamWidth, labeled, lowerCased, numThreads,
                    partial, scorePath);
        else
            parseCoNLLFileParallel(inputFile, outputFile, rootFirst, beamWidth, lowerCased, numThreads, partial,
                    scorePath);
    }

    private void parseCoNLLFileNoParallel(String inputFile, String outputFile, boolean rootFirst, int beamWidth,
                                          boolean labeled, boolean lowerCased, int numOfThreads, boolean partial,
                                          String scorePath) throws Exception {
        CoNLLReader reader = new CoNLLReader(inputFile);
        boolean addScore = false;
        if (scorePath.trim().length() > 0)
            addScore = true;
        ArrayList<Float> scoreList = new ArrayList<>();
        long start = System.currentTimeMillis();
        int allArcs = 0;
        int size = 0;
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile + ".tmp"));
        int dataCount = 0;
        while (true) {
            ArrayList<GoldConfiguration> data = reader.readData(15000, true, labeled, rootFirst, lowerCased, maps);
            size += data.size();
            if (data.size() == 0)
                break;
            for (GoldConfiguration goldConfiguration : data) {
                dataCount++;
                if (dataCount % 100 == 0)
                    System.out.print(dataCount + " ... ");
                Configuration bestParse;
                if (partial)
                    bestParse = parsePartial(goldConfiguration, goldConfiguration.getSentence(), rootFirst, beamWidth
                            , numOfThreads);
                else bestParse = parse(goldConfiguration.getSentence(), rootFirst, beamWidth, numOfThreads);
                int[] words = goldConfiguration.getSentence().getWords();
                allArcs += words.length - 1;
                if (addScore)
                    scoreList.add(bestParse.score / bestParse.sentence.size());
                StringBuilder finalOutput = new StringBuilder();
                for (int i = 0; i < words.length; i++) {
                    int w = i + 1;
                    int head = bestParse.state.getHead(w);
                    int dep = bestParse.state.getDependency(w);
                    if (w == bestParse.state.rootIndex && !rootFirst)
                        continue;
                    if (head == bestParse.state.rootIndex)
                        head = 0;
                    String label = head == 0 ? maps.rootString : maps.revWords[dep];
                    String output = head + "\t" + label + "\n";
                    finalOutput.append(output);
                }
                finalOutput.append("\n");
                writer.write(finalOutput.toString());
            }
        }
        System.out.print("\n");
        long end = System.currentTimeMillis();
        float each = (1.0f * (end - start)) / size;
        float eacharc = (1.0f * (end - start)) / allArcs;
        writer.close();
        DecimalFormat format = new DecimalFormat("##.00");
        System.out.println(format.format(eacharc) + " ms for each arc!");
        System.out.println(format.format(each) + " ms for each sentence!");
        System.out.println();
        BufferedReader gReader = new BufferedReader(new FileReader(inputFile));
        BufferedReader pReader = new BufferedReader(new FileReader(outputFile + ".tmp"));
        BufferedWriter pWriter = new BufferedWriter(new FileWriter(outputFile));
        String line;
        while ((line = pReader.readLine()) != null) {
            String gLine = gReader.readLine();
            if (line.trim().length() > 0) {
                while (gLine.trim().length() == 0)
                    gLine = gReader.readLine();
                String[] ps = line.split("\t");
                String[] gs = gLine.split("\t");
                gs[6] = ps[0];
                gs[7] = ps[1];
                StringBuilder output = new StringBuilder();
                for (String g : gs) {
                    output.append(g).append("\t");
                }
                pWriter.write(output.toString().trim() + "\n");
            } else {
                pWriter.write("\n");
            }
        }
        gReader.close();
        pReader.close();
        pWriter.close();
        if (addScore) {
            BufferedWriter scoreWriter = new BufferedWriter(new FileWriter(scorePath));
            for (Float aFloat : scoreList) scoreWriter.write(aFloat + "\n");
            scoreWriter.flush();
            scoreWriter.close();
        }
    }

    public void parseTaggedFile(String inputFile, String outputFile, boolean rootFirst, int beamWidth,
                                boolean lowerCased, String separator, int numOfThreads) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
        long start = System.currentTimeMillis();
        ExecutorService executor = Executors.newFixedThreadPool(numOfThreads);
        CompletionService<Pair<String, Integer>> pool = new ExecutorCompletionService<>(executor);
        String line;
        int count = 0;
        int lineNum = 0;
        while ((line = reader.readLine()) != null) {
            pool.submit(new ParseTaggedThread(lineNum++, line, separator, rootFirst, lowerCased, maps, beamWidth,
                    this));
            if (lineNum % 1000 == 0) {
                String[] outs = new String[lineNum];
                for (int i = 0; i < lineNum; i++) {
                    count++;
                    if (count % 100 == 0)
                        System.out.print(count + "...");
                    Pair<String, Integer> result = pool.take().get();
                    outs[result.second] = result.first;
                }
                for (int i = 0; i < lineNum; i++) {
                    if (outs[i].length() > 0) {
                        writer.write(outs[i]);
                    }
                }
                lineNum = 0;
            }
        }
        if (lineNum > 0) {
            String[] outs = new String[lineNum];
            for (int i = 0; i < lineNum; i++) {
                count++;
                if (count % 100 == 0)
                    System.out.print(count + "...");
                Pair<String, Integer> result = pool.take().get();
                outs[result.second] = result.first;
            }
            for (int i = 0; i < lineNum; i++) {
                if (outs[i].length() > 0) {
                    writer.write(outs[i]);
                }
            }
        }
        long end = System.currentTimeMillis();
        System.out.println("\n" + (end - start) + " ms");
        reader.close();
        writer.close();
        System.out.println("done!");
    }

    private void parseCoNLLFileParallel(String inputFile, String outputFile, boolean rootFirst, int beamWidth,
                                        boolean lowerCased, int numThreads, boolean partial, String scorePath) throws Exception {
        boolean addScore = false;
        if (scorePath.trim().length() > 0) {
            addScore = true;
        }
        ArrayList<Float> scoreList = new ArrayList<>();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<Pair<Configuration, Integer>> pool = new ExecutorCompletionService<>(executor);
        long start = System.currentTimeMillis();
        int allArcs = 0;
        int size = 0;
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile + ".tmp"));
        int dataCount = 0;
        CoNLLReader reader = new CoNLLReader(inputFile);
        ArrayList<GoldConfiguration> data = reader.readData(15000, true, true, rootFirst, lowerCased, maps);
        while (data.size() != 0) {
            size += data.size();
            int index = 0;
            Configuration[] confs = new Configuration[data.size()];
            for (GoldConfiguration goldConfiguration : data) {
                ParseThread thread = new ParseThread(index, bClassifier, classifier, dependencyRelations, featureLength,
                        goldConfiguration.getSentence(), rootFirst, beamWidth, goldConfiguration, partial);
                pool.submit(thread);
                index++;
            }
            System.out.println("sentences parsed:");
            for (int i = 0; i < confs.length; i++) {
                dataCount++;
                if (dataCount % 100 == 0) {
                    System.out.print(dataCount + " ... ");
                }
                Pair<Configuration, Integer> configurationIntegerPair = pool.take().get();
                confs[configurationIntegerPair.second] = configurationIntegerPair.first;
            }
            System.out.println();
            for (int j = 0; j < confs.length; j++) {
                Configuration bestParse = confs[j];
                if (addScore) {
                    scoreList.add(bestParse.score / bestParse.sentence.size());
                }
                int sentenceWords = data.get(j).getSentence().getWords().length;
                allArcs += sentenceWords - 1;
                StringBuilder finalOutput = new StringBuilder();
                for (int i = 1; i <= sentenceWords; i++) {
                    if (i == bestParse.state.rootIndex && !rootFirst) {
                        continue;
                    }
                    int head = bestParse.state.getHead(i);
                    if (head == bestParse.state.rootIndex) {
                        head = 0;
                    }
                    int dep = bestParse.state.getDependency(i);
                    String label = head == 0 ? maps.rootString : maps.revWords[dep];
                    finalOutput.append(head).append("\t").append(label).append("\n");
                }
                finalOutput.append("\n");
                writer.write(finalOutput.toString());
            }
            data = reader.readData(15000, true, true, rootFirst, lowerCased, maps);
        }
        long end = System.currentTimeMillis();
        float eachSent = (1.0f * (end - start)) / size;
        float eachArc = (1.0f * (end - start)) / allArcs;
        writer.close();
        DecimalFormat format = new DecimalFormat("##.00");
        System.out.println(format.format(eachArc) + " ms for each arc!");
        System.out.println(format.format(eachSent) + " ms for each sentence!");
        System.out.println();
        BufferedReader gReader = new BufferedReader(new FileReader(inputFile));
        BufferedReader pReader = new BufferedReader(new FileReader(outputFile + ".tmp"));
        BufferedWriter pwriter = new BufferedWriter(new FileWriter(outputFile));
        String line;
        while ((line = pReader.readLine()) != null) {
            String gLine = gReader.readLine();
            if (line.trim().length() > 0) {
                while (gLine.trim().length() == 0)
                    gLine = gReader.readLine();
                String[] ps = line.split("\t");
                String[] gs = gLine.split("\t");
                gs[6] = ps[0];
                gs[7] = ps[1];
                StringBuilder output = new StringBuilder();
                for (String g : gs) {
                    output.append(g).append("\t");
                }
                pwriter.write(output.toString().trim() + "\n");
            } else {
                pwriter.write("\n");
            }
        }
        gReader.close();
        pReader.close();
        pwriter.close();
        Files.deleteIfExists(Path.of(outputFile + ".tmp"));
        if (addScore) {
            BufferedWriter scoreWriter = new BufferedWriter(new FileWriter(scorePath));
            for (Float aFloat : scoreList) scoreWriter.write(aFloat + "\n");
            scoreWriter.flush();
            scoreWriter.close();
        }
    }

    public void shutDownLiveThreads() {
        boolean isTerminated = executor.isTerminated();
        while (!isTerminated) {
            executor.shutdownNow();
            isTerminated = executor.isTerminated();
        }
    }
}