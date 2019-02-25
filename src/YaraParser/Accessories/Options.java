package YaraParser.Accessories;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;

public class Options implements Serializable {
    public boolean train;
    public boolean parseTaggedFile;
    public boolean parseConllFile;
    public int beamWidth;
    public boolean rootFirst;
    public boolean showHelp;
    public boolean labeled;
    public String inputFile;
    public String outputFile;
    public String devPath;
    public int trainingIter;
    public boolean evaluate;
    public boolean parsePartialConll;
    public String scorePath;
    public String clusterFile;

    public String modelFile;
    public boolean lowercase;
    public boolean useExtendedFeatures;
    public boolean useExtendedWithBrownClusterFeatures;
    public boolean useMaxViol;
    public boolean useDynamicOracle;
    public boolean useRandomOracleSelection;
    public String separator;
    public int numOfThreads;

    public String goldFile;

    public HashSet<String> punctuations;
    public String predFile;

    public int partialTrainingStartingIteration;

    public Options() {
        showHelp = false;
        train = false;
        parseConllFile = false;
        parseTaggedFile = false;
        beamWidth = 64;
        rootFirst = false;
        modelFile = "";
        outputFile = "";
        inputFile = "";
        devPath = "";
        scorePath = "";
        separator = "_";
        clusterFile = "";
        labeled = true;
        lowercase = false;
        useExtendedFeatures = true;
        useMaxViol = true;
        useDynamicOracle = true;
        useRandomOracleSelection = false;
        trainingIter = 20;
        evaluate = false;
        numOfThreads = 8;
        useExtendedWithBrownClusterFeatures = false;
        parsePartialConll = false;

        partialTrainingStartingIteration = 3;

        punctuations = new HashSet<>();
        punctuations.add("#");
        punctuations.add("''");
        punctuations.add("(");
        punctuations.add(")");
        punctuations.add("[");
        punctuations.add("]");
        punctuations.add("{");
        punctuations.add("}");
        punctuations.add("\"");
        punctuations.add(",");
        punctuations.add(".");
        punctuations.add(":");
        punctuations.add("``");
        punctuations.add("-LRB-");
        punctuations.add("-RRB-");
        punctuations.add("-LSB-");
        punctuations.add("-RSB-");
        punctuations.add("-LCB-");
        punctuations.add("-RCB-");
        punctuations.add("!");
        punctuations.add("$");
        punctuations.add("?");
    }

    public static void showHelp() {

        String output = "© Yara YaraParser.Parser \n" +
                "\u00a9 Copyright 2014, Yahoo! Inc.\n" +
                "\u00a9 Licensed under the terms of the Apache License 2.0. See LICENSE file at the project root for terms." +
                "http://www.apache.org/licenses/LICENSE-2.0\n" +
                "\n" +
                "Usage:\n" +
                "* Train a parser:\n" +
                "\tjava -jar YaraParser.jar train -train-file [train-file] -dev [dev-file] -model [model-file] -punc [punc-file]\n" +
                "\t** The model for each iteration is with the pattern [model-file]_iter[iter#]; e.g. mode_iter2\n" +
                "\t** [punc-file]: File contains list of pos tags for punctuations in the treebank, each in one line\n" +
                "\t** Other options\n" +
                "\t \t -cluster [cluster-file] Brown cluster file: at most 4096 clusters are supported by the parser (default: empty)\n\t\t\t the format should be the same as https://github.com/percyliang/brown-cluster/blob/master/output.txt \n" +
                "\t \t beam:[beam-width] (default:64)\n" +
                "\t \t iter:[training-iterations] (default:20)\n" +
                "\t \t unlabeled (default: labeled parsing, unless explicitly put `unlabeled')\n" +
                "\t \t lowercase (default: case-sensitive words, unless explicitly put 'lowercase')\n" +
                "\t \t basic (default: use extended feature set, unless explicitly put 'basic')\n" +
                "\t \t early (default: use max violation update, unless explicitly put `early' for early update)\n" +
                "\t \t static (default: use dynamic oracles, unless explicitly put `static' for static oracles)\n" +
                "\t \t random (default: choose maximum scoring oracle, unless explicitly put `random' for randomly choosing an oracle)\n" +
                "\t \t nt:[#_of_threads] (default:8)\n" +
                "\t \t pt:[#partail_training_starting_iteration] (default:3; shows the starting iteration for considering partial trees)\n" +
                "\t \t root_first (default: put ROOT in the last position, unless explicitly put 'root_first')\n\n" +
                "* Parse a CoNLL'2006 file:\n" +
                "\tjava -jar YaraParser.jar parse_conll -input [test-file] -out [output-file] -model [model-file] nt:[#_of_threads (optional -- default:8)] \n" +
                "\t** The test file should have the conll 2006 format\n" +
                "\t** Optional: -score [score file] averaged score of each output parse tree in a file\n\n" +
                "* Parse a tagged file:\n" +
                "\tjava -jar YaraParser.jar parse_tagged -input [test-file] -out [output-file]  -model [model-file] nt:[#_of_threads (optional -- default:8)] \n" +
                "\t** The test file should have each sentence in line and word_tag pairs are space-delimited\n" +
                "\t** Optional:  -delim [delim] (default is _)\n" +
                "\t \t Example: He_PRP is_VBZ nice_AJ ._.\n\n" +
                "* Parse a CoNLL'2006 file with partial gold trees:\n" +
                "\tjava -jar YaraParser.jar parse_partial -input [test-file] -out [output-file] -model [model-file] nt:[#_of_threads (optional -- default:8)] \n" +
                "\t** The test file should have the conll 2006 format; each word that does not have a parent, should have a -1 parent-index" +
                "\t** Optional: -score [score file] averaged score of each output parse tree in a file\n\n" +
                "* Evaluate a Conll file:\n" +
                "\tjava -jar YaraParser.jar eval -gold [gold-file] -parse [parsed-file]  -punc [punc-file]\n" +
                "\t** [punc-file]: File contains list of pos tags for punctuations in the treebank, each in one line\n" +
                "\t** Both files should have conll 2006 format\n";
        System.out.println(output);
    }

    public static Options processArgs(String[] args) throws Exception {
        Options options = new Options();

        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--help") || args[i].equals("-h") || args[i].equals("-help"))
                options.showHelp = true;
            else if (args[i].equals("train"))
                options.train = true;
            else if (args[i].equals("parse_conll"))
                options.parseConllFile = true;
            else if (args[i].equals("parse_partial"))
                options.parsePartialConll = true;
            else if (args[i].equals("eval"))
                options.evaluate = true;
            else if (args[i].equals("parse_tagged"))
                options.parseTaggedFile = true;
            else if (args[i].equals("-train-file") || args[i].equals("-input"))
                options.inputFile = args[i + 1];
            else if (args[i].equals("-punc"))
                options.changePunc(args[i + 1]);
            else if (args[i].equals("-model"))
                options.modelFile = args[i + 1];
            else if (args[i].startsWith("-dev"))
                options.devPath = args[i + 1];
            else if (args[i].equals("-gold"))
                options.goldFile = args[i + 1];
            else if (args[i].startsWith("-parse"))
                options.predFile = args[i + 1];
            else if (args[i].startsWith("-cluster")) {
                options.clusterFile = args[i + 1];
                options.useExtendedWithBrownClusterFeatures = true;
            } else if (args[i].startsWith("-out"))
                options.outputFile = args[i + 1];
            else if (args[i].startsWith("-delim"))
                options.separator = args[i + 1];
            else if (args[i].startsWith("beam:"))
                options.beamWidth = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
            else if (args[i].startsWith("nt:"))
                options.numOfThreads = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
            else if (args[i].startsWith("pt:"))
                options.partialTrainingStartingIteration = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
            else if (args[i].equals("unlabeled"))
                options.labeled = Boolean.parseBoolean(args[i]);
            else if (args[i].equals("lowercase"))
                options.lowercase = Boolean.parseBoolean(args[i]);
            else if (args[i].startsWith("-score"))
                options.scorePath = args[i + 1];
            else if (args[i].equals("basic"))
                options.useExtendedFeatures = false;
            else if (args[i].equals("early"))
                options.useMaxViol = false;
            else if (args[i].equals("static"))
                options.useDynamicOracle = false;
            else if (args[i].equals("random"))
                options.useRandomOracleSelection = true;
            else if (args[i].equals("root_first"))
                options.rootFirst = true;
            else if (args[i].startsWith("iter:"))
                options.trainingIter = Integer.parseInt(args[i].substring(args[i].lastIndexOf(":") + 1));
        }

        if (options.train || options.parseTaggedFile || options.parseConllFile)
            options.showHelp = false;

        return options;
    }

    public static ArrayList<Options> getAllPossibleOptions(Options option) {
        ArrayList<Options> options = new ArrayList<>();
        options.add(option);

        ArrayList<Options> tmp = new ArrayList<>();

        for (Options opt : options) {
            Options o1 = opt.clone();
            o1.labeled = true;

            Options o2 = opt.clone();
            o2.labeled = false;
            tmp.add(o1);
            tmp.add(o2);
        }

        options = tmp;
        tmp = new ArrayList<>();


        for (Options opt : options) {
            Options o1 = opt.clone();
            o1.lowercase = true;

            Options o2 = opt.clone();
            o2.lowercase = false;
            tmp.add(o1);
            tmp.add(o2);
        }

        options = tmp;
        tmp = new ArrayList<>();

        for (Options opt : options) {
            Options o1 = opt.clone();
            o1.useExtendedFeatures = true;

            Options o2 = opt.clone();
            o2.useExtendedFeatures = false;
            tmp.add(o1);
            tmp.add(o2);
        }

        options = tmp;
        tmp = new ArrayList<>();

        for (Options opt : options) {
            Options o1 = opt.clone();
            o1.useDynamicOracle = true;

            Options o2 = opt.clone();
            o2.useDynamicOracle = false;
            tmp.add(o1);
            tmp.add(o2);
        }

        options = tmp;
        tmp = new ArrayList<>();

        for (Options opt : options) {
            Options o1 = opt.clone();
            o1.useMaxViol = true;

            Options o2 = opt.clone();
            o2.useMaxViol = false;
            tmp.add(o1);
            tmp.add(o2);
        }

        options = tmp;
        tmp = new ArrayList<>();

        for (Options opt : options) {
            Options o1 = opt.clone();
            o1.useRandomOracleSelection = true;

            Options o2 = opt.clone();
            o2.useRandomOracleSelection = false;
            tmp.add(o1);
            tmp.add(o2);
        }

        options = tmp;
        tmp = new ArrayList<>();


        for (Options opt : options) {
            Options o1 = opt.clone();
            o1.rootFirst = true;

            Options o2 = opt.clone();
            o2.rootFirst = false;
            tmp.add(o1);
            tmp.add(o2);
        }

        options = tmp;
        return options;
    }

    public void changePunc(String puncPath) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(puncPath));

        punctuations = new HashSet<>();
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.length() > 0)
                punctuations.add(line.split(" ")[0].trim());
        }
    }

    public String toString() {
        if (train) {
            StringBuilder builder = new StringBuilder();
            builder.append("train file: ").append(inputFile).append("\n");
            builder.append("dev file: ").append(devPath).append("\n");
            builder.append("cluster file: ").append(clusterFile).append("\n");
            builder.append("beam width: ").append(beamWidth).append("\n");
            builder.append("rootFirst: ").append(rootFirst).append("\n");
            builder.append("labeled: ").append(labeled).append("\n");
            builder.append("lower-case: ").append(lowercase).append("\n");
            builder.append("extended features: ").append(useExtendedFeatures).append("\n");
            builder.append("extended with brown cluster features: ").append(useExtendedWithBrownClusterFeatures).append("\n");
            builder.append("updateModel: ").append(useMaxViol ? "max violation" : "early").append("\n");
            builder.append("oracle: ").append(useDynamicOracle ? "dynamic" : "static").append("\n");
            if (useDynamicOracle)
                builder.append("oracle selection: ").append(!useRandomOracleSelection ? "latent max" : "random").append("\n");

            builder.append("training-iterations: ").append(trainingIter).append("\n");
            builder.append("number of threads: ").append(numOfThreads).append("\n");
            builder.append("partial training starting iteration: ").append(partialTrainingStartingIteration).append("\n");
            return builder.toString();
        } else if (parseConllFile) {
            return ("parse conll" + "\n") +
                    "input file: " + inputFile + "\n" +
                    "output file: " + outputFile + "\n" +
                    "model file: " + modelFile + "\n" +
                    "score file: " + scorePath + "\n" +
                    "number of threads: " + numOfThreads + "\n";
        } else if (parseTaggedFile) {
            return ("parse  tag file" + "\n") +
                    "input file: " + inputFile + "\n" +
                    "output file: " + outputFile + "\n" +
                    "model file: " + modelFile + "\n" +
                    "score file: " + scorePath + "\n" +
                    "number of threads: " + numOfThreads + "\n";
        } else if (parsePartialConll) {
            return ("parse partial conll" + "\n") +
                    "input file: " + inputFile + "\n" +
                    "output file: " + outputFile + "\n" +
                    "score file: " + scorePath + "\n" +
                    "model file: " + modelFile + "\n" +
                    "labeled: " + labeled + "\n" +
                    "number of threads: " + numOfThreads + "\n";
        } else if (evaluate) {
            return ("Evaluate" + "\n") +
                    "gold file: " + goldFile + "\n" +
                    "parsed file: " + predFile + "\n";
        }
        return "";
    }

    public Options clone() {
        Options options = new Options();
        options.train = train;
        options.labeled = labeled;
        options.trainingIter = trainingIter;
        options.useMaxViol = useMaxViol;
        options.beamWidth = beamWidth;
        options.devPath = devPath;
        options.evaluate = evaluate;
        options.goldFile = goldFile;
        options.inputFile = inputFile;
        options.lowercase = lowercase;
        options.numOfThreads = numOfThreads;
        options.outputFile = outputFile;
        options.useDynamicOracle = useDynamicOracle;
        options.modelFile = modelFile;
        options.rootFirst = rootFirst;
        options.parseConllFile = parseConllFile;
        options.parseTaggedFile = parseTaggedFile;
        options.predFile = predFile;
        options.showHelp = showHelp;
        options.separator = separator;
        options.useExtendedFeatures = useExtendedFeatures;
        options.parsePartialConll = parsePartialConll;
        options.partialTrainingStartingIteration = partialTrainingStartingIteration;
        return options;
    }
}
