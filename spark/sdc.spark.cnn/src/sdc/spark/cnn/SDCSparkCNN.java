package sdc.spark.cnn;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.deeplearning4j.spark.api.RDDTrainingApproach;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;


public class SDCSparkCNN {
    protected static final Logger log = LoggerFactory.getLogger(SDCSparkCNN.class);

    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 30;

    @Parameter(names = "-epochs", description = "Number of epochs for training")
    private int epochs = 50;

    @Parameter(names = "-dirPath", description = "Image dataset path")
    private String dirPath = "sdcdata/sdcdata1000c/"; // 1000 random examples per label, relatively clean

    @Parameter(names = "-modelType", description = "modelType for the CNN")
    private String modelType = "AlexNet";

    @Parameter(names = "-height", description = "height for input layer")
    private int height = 100;

    @Parameter(names = "-width", description = "width for input layer")
    private int width = 100;

    @Parameter(names = "-iterations", description = "iterations")
    private int iterations = 1;

    @Parameter(names = "-averagingFrequency", description = "averagingFrequency")
    private int averagingFrequency = 5;

    @Parameter(names = "-numLabels", description = "number of labels")
    private int numLabels = 3;

    @Parameter(names = "-channels", description = "number of channels for images")
    private int channels = 3;


    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static double splitTrainTest = 0.8;
    protected static boolean save = false;
    protected static long trainTime, testTime, startTime, endTime, totalTime;

    public static void main(String[] args) throws Exception {
        totalTime = System.currentTimeMillis();
        new SDCSparkCNN().run(args);
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

        System.out.println("****** Parameter Values ******");
        System.out.println("useSparkLocal: " + useSparkLocal);
        System.out.println("modelType: " + modelType);
        System.out.println("dirPath: " + dirPath);
        System.out.println("numLabels: " + numLabels);
        System.out.println("channels: " + channels);
        System.out.println("epochs: " + epochs);
        System.out.println("iterations: " + iterations);
        System.out.println("height: " + height);
        System.out.println("width: " + width);
        System.out.println("batchSizePerWorker: " + batchSizePerWorker);

        // Configure Spark and training master
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[2]");
        } else {
            //sparkConf.setMaster("spark://OmerMBP.local:7077");
        }
        sparkConf.setAppName("SDC CNN on Spark");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
                .averagingFrequency(averagingFrequency)
                .workerPrefetchNumBatches(2)
                .batchSizePerWorker(batchSizePerWorker)
                //.rddTrainingApproach(RDDTrainingApproach.Direct)
                //.storageLevel(StorageLevel.MEMORY_ONLY_SER())
                .build();

        // Load images from user.home/dirPath
        System.out.println("****** Loading images ******");
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(System.getProperty("user.home"), dirPath);
        FileSplit fileSplit = new FileSplit(mainPath, new String[] {"jpg"}, rng);
        InputSplit[] inputSplit = fileSplit.sample(null, splitTrainTest, 1 - splitTrainTest);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        // Scaler for normalization
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator dataIter;

        // Create training data JavaRDD
        recordReader.initialize(trainData);
        List<String> labelNames = recordReader.getLabels();
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSizePerWorker, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        List<DataSet> trainDataList = new ArrayList<>();
        while (dataIter.hasNext()) {
            trainDataList.add(dataIter.next());
        }
        JavaRDD<DataSet> trainDataRDD = sc.parallelize(trainDataList);

        // Create test data JavaRDD
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSizePerWorker, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        List<DataSet> testDataList = new ArrayList<>();
        while (dataIter.hasNext()) {
            testDataList.add(dataIter.next());
        }
        JavaRDD<DataSet> testDataRDD = sc.parallelize(testDataList);

        MultiLayerNetwork network;
        switch (modelType) {
            case "LeNet":
                network = LeNet();
                break;
            case "AlexNet":
                network = AlexNet();
                break;
            case "VGGNet":
                network = VGGNet();
                break;
            default:
                throw new InvalidInputTypeException("Incorrect model provided.");
        }

        System.out.println("****** Building network ******");
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, network, tm);

        // Perform training
        System.out.println("****** Training network ******");
        startTime = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++) {
            sparkNet.fit(trainDataRDD);
            System.out.println("****** Completed epoch " + i + " ******");
        }
        endTime = System.currentTimeMillis();
        trainTime = endTime - startTime;

        // Perform evaluation (distributed)
        System.out.println("****** Evaluating network ******");
        startTime = System.currentTimeMillis();
        Evaluation evaluation = sparkNet.evaluate(testDataRDD);
        endTime = System.currentTimeMillis();
        testTime = endTime - startTime;
        System.out.println("Labels: ");
        for (int x = 0; x < labelNames.size(); x++){
            System.out.println(x + ": " + labelNames.get(x));
        }
        System.out.println("\nEvaluation:");
        System.out.println(evaluation.stats());
        System.out.println("Training runtime: " + timeToString(trainTime));
        System.out.println("Testing runtime: " + timeToString(testTime));
        totalTime = System.currentTimeMillis() - totalTime;
        System.out.println("Total runtime: " + timeToString(totalTime));

        if (save) { //TODO: fix saving CNN
            System.out.println("****** Saving network ******");
            //String basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/");
            //NetSaverLoaderUtils.saveNetworkAndParameters(network, basePath);
            //NetSaverLoaderUtils.saveUpdators(network, basePath);
        }

        // Delete the temp training files
        tm.deleteTempFiles(sc);
    }

    public static String timeToString(long time){
        int min = (int)TimeUnit.MILLISECONDS.toMinutes(time);
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
        /**
         * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
         * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
         **/
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
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

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