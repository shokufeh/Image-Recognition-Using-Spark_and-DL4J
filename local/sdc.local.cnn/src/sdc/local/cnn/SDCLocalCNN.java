package sdc.local.cnn;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;


public class SDCLocalCNN {

    protected static final Logger log = LoggerFactory.getLogger(SDCLocalCNN.class);

    @Parameter(names = "-useUIServer", description = "use spark UI server", arity = 1)
    private boolean useUIServer = true;

    @Parameter(names = "-epochs", description = "epochs for the CNN")
    private int epochs = 50;

    @Parameter(names = "-modelType", description = "modelType for the CNN")
    private String modelType = "LeNet";

    @Parameter(names = "-height", description = "height for input layer")
    private int height = 100;

    @Parameter(names = "-width", description = "width for input layer")
    private int width = 100;

    @Parameter(names = "-iterations", description = "iterations")
    private int iterations = 1;

    @Parameter(names = "-dirPath", description = "main path of dir containing image data")
    private String dirPath = "sdcdata/sdcdata1000c/";

    @Parameter(names = "-maxExamples", description = "max number of examples")
    private int maxExamples = 300;

    @Parameter(names = "-saveNetwork", description = "save network to disk")
    private boolean saveNetwork = false;

    @Parameter(names = "-batchSize", description = "batch size")
    private int batchSize = 18;

    @Parameter(names = "-numLabels", description = "number of labels")
    private int numLabels = 3;

    @Parameter(names = "-channels", description = "number of channels for images")
    private int channels = 3;

    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;
    protected static double splitTrainTest = 0.8;
    protected static int nCores = 8;
    protected static long trainTime, testTime, startTime, endTime, totalTime;


    public static void main(String[] args) throws Exception {
        totalTime = System.currentTimeMillis();
        new SDCLocalCNN().run(args);
    }

    public void run(String[] args) throws Exception {

        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            jcmdr.usage();
            try { Thread.sleep(500); } catch (Exception e2) { }
            throw e;
        }

        System.out.println("****** Parameter Valies ******");
        System.out.println("useUIServer: " + useUIServer);
        System.out.println("modelType: " + modelType);
        System.out.println("dirPath: " + dirPath);
        System.out.println("maxExamples: " + maxExamples);
        System.out.println("numLabels: " + numLabels);
        System.out.println("channels: " + channels);
        System.out.println("epochs: " + epochs);
        System.out.println("iterations: " + iterations);
        System.out.println("height: " + height);
        System.out.println("width: " + width);
        System.out.println("batchSize: " + batchSize);

        // Load images from user.home/dirPath
        System.out.println("****** Loading images ******");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(System.getProperty("user.home"), dirPath);
        FileSplit fileSplit = new FileSplit(mainPath, new String[] {"jpg"}, rng);
        RandomPathFilter pathFilter = new RandomPathFilter(rng, new String[] {"jpg"}, maxExamples);
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        MultiLayerNetwork network;
        switch (modelType) {
            case "LeNet":
                network = LeNet();
                break;
            case "AlexNet":
                network = AlexNet();
                break;
            case "VGGNet":
                network = VGGNet(); //TODO: fix VGGNet implementation
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }
        network.init();

        // attach listener for UIServer during training
        if (useUIServer) {
            UIServer uiServer = UIServer.getInstance();
            //StatsStorage statsStorage = new InMemoryStatsStorage();
            StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("user.home"),"UIData/UIdata.dat"));
            uiServer.attach(statsStorage);
            network.setListeners(new StatsListener(statsStorage));
        } else {
            network.setListeners(new ScoreIterationListener(listenerFreq));
        }

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;
        recordReader.initialize(trainData, null);
        List<String> labelNames = recordReader.getLabels();
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);

        System.out.println("****** Training network ******");
        startTime = System.currentTimeMillis();
        network.fit(trainIter);
        endTime = System.currentTimeMillis();
        trainTime = endTime - startTime;

        System.out.println("****** Evaluating network ******");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        startTime = System.currentTimeMillis();
        Evaluation evaluation = network.evaluate(dataIter);
        endTime = System.currentTimeMillis();
        testTime = endTime - startTime;
        System.out.println("Labels: ");
        for (int x = 0; x < labelNames.size(); x++){
            System.out.println(x + ": " + labelNames.get(x));
        }
        System.out.println("\nEvaluation:");
        System.out.println(evaluation.stats(true));
        System.out.println("Training time: " + timeToString(trainTime));
        System.out.println("Testing time: " + timeToString(testTime));
        totalTime = System.currentTimeMillis() - totalTime;

        // Example on how to get predict results with trained model
        dataIter.reset();
        DataSet testDataSet = dataIter.next();
        String expectedResult = testDataSet.getLabelName(0);
        List<String> predict = network.predict(testDataSet);
        String modelResult = predict.get(0);
        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n");

        if (saveNetwork) {
            System.out.println("****** Saving network ******");
            //String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
            //NetSaverLoaderUtils.saveNetworkAndParameters(network, basePath);
            //NetSaverLoaderUtils.saveUpdators(network, basePath);
        }
    }

    public static String timeToString(long time){
        int min = (int) TimeUnit.MILLISECONDS.toMinutes(time);
        int sec = (int)(TimeUnit.MILLISECONDS.toSeconds(time) - TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(time)));
        int ms = (int)time;
        return min + " min, " + sec + " sec, " + ms + " ms";
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

    public MultiLayerNetwork LeNet() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .regularization(false).l2(0.005) // tried 0.0001, 0.0005
            .activation(Activation.RELU)
            .learningRate(0.0001) // tried 0.00001, 0.00005, 0.000001
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.RMSPROP).momentum(0.9)
            .list()
            .layer(0, convInit("cnn1", channels, 50 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
            .layer(1, maxPool("maxpool1", new int[]{2,2}))
            .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
            .layer(3, maxPool("maxool2", new int[]{2,2}))
            .layer(4, new DenseLayer.Builder().nOut(500).build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true).pretrain(false)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        return new MultiLayerNetwork(conf);

    }

    public MultiLayerNetwork AlexNet() {
        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(new NormalDistribution(0.0, 0.01))
            .activation(Activation.RELU)
            .updater(Updater.NESTEROVS)
            .iterations(iterations)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-2)
            .biasLearningRate(1e-2*2)
            .learningRateDecayPolicy(LearningRatePolicy.Step)
            .lrPolicyDecayRate(0.1)
            .lrPolicySteps(100000)
            .regularization(true)
            .l2(5 * 1e-4)
            .momentum(0.9)
            .miniBatch(false)
            .list()
            .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
            .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
            .layer(2, maxPool("maxpool1", new int[]{3,3}))
            .layer(3, conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
            .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
            .layer(5, maxPool("maxpool2", new int[]{3,3}))
            .layer(6,conv3x3("cnn3", 384, 0))
            .layer(7,conv3x3("cnn4", 384, nonZeroBias))
            .layer(8,conv3x3("cnn5", 256, nonZeroBias))
            .layer(9, maxPool("maxpool3", new int[]{3,3}))
            .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
            .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
            .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build())
            .backprop(true)
            .pretrain(false)
            .setInputType(InputType.convolutional(height, width, channels))
            .build();

        return new MultiLayerNetwork(conf);

    }

    public MultiLayerNetwork VGGNet() {

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) //
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(Updater.NESTEROVS)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-1)
                .learningRateScoreBasedDecayRate(1e-1)
                .regularization(true)
                .l2(5 * 1e-4)
                .momentum(0.9)
                .list()
                .layer(0, convInit("cnn1", channels, 64, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, convInit("cnn2", channels, 128, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxpool2", new int[]{2,2}))
                .layer(4, convInit("cnn3", channels, 256, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(5, convInit("cnn4", channels, 256, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(6, maxPool("maxpool3", new int[]{2,2}))
                .layer(7, convInit("cnn5", channels, 512, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(8, convInit("cnn6", channels, 512, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(9, maxPool("maxpool4", new int[]{2,2}))
                .layer(10, convInit("cnn7", channels, 512, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(11, convInit("cnn8", channels, 512, new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(12, maxPool("maxpool5", new int[]{2,2}))
                .layer(13, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(14, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(15, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);
    }
}