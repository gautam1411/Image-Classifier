/***
 *
 *
 *  ConvNet Class  defines Architecture of Convolutional Neural Network
 *
 */

import java.util.Vector;

public class ConvNet{


    private boolean debugCNN = true;
    private int countClasses;

    private Convolution conv1;
    private Pooling maxPool1;
    private Convolution conv2;
    private Pooling maxPool2;
    private Convolution conv3;
    private Pooling maxPool3;
    private FlatLayer flat;
    private OutputLayer out;
    private int bestTune;


   public ConvNet( Vector<Vector<Double>> inputFeatureVectors , int hyperparameters, boolean debugSwitch){

       debugCNN     = debugSwitch;

       conv1        = new Convolution(inputFeatureVectors, hyperparameters, debugCNN);    // Conv-1
       maxPool1     = new Pooling(conv1, debugCNN);                                       // Pool-1
       conv2        = new Convolution(maxPool1, hyperparameters,  debugCNN);              // Conv-2
       maxPool2     = new Pooling(conv2, debugCNN);                                       // Pool-2
       //conv3        = new Convolution(maxPool2, hyperparameters,debugCNN);              // Conv-3
      // maxPool3     = new Pooling(conv3, debugCNN);                                     // Pool-3
       flat         = new FlatLayer(maxPool2, debugCNN);                                  // Flat Fully Connected Layer
       out          = new OutputLayer(flat, hyperparameters, debugCNN);                   // Output Layer

   }


   public int trainCNN( Vector<Vector<Double>> trainFeatureVectors) {
	   
	   out.resetCountCorrect();
       int errorCount = 0;
	   
       for (int trainingIpNum = 0; trainingIpNum < trainFeatureVectors.size(); trainingIpNum++) {

           Vector<Double> trainFeatureVector = trainFeatureVectors.get(trainingIpNum);

           conv1.train(trainFeatureVector);
           maxPool1.train(conv1);
           conv2.train(maxPool1);
           maxPool2.train(conv2);
           //conv3.train(maxPool2);
           //maxPool3.train(conv3);

           flat.trainwithDropOut(maxPool2);
           out.train(flat);
           out.backpropagate();
           flat.backpropagate(out);

          // maxPool3.backpropagate(flat);
          // conv3.backpropagate(maxPool3);

           maxPool2.backpropagate(flat);
           conv2.backpropagate(maxPool2);

           maxPool1.backpropagate(conv2);
           conv1.backpropagate(maxPool1);

           if (debugCNN) {

               conv1.printActivationMaps();
               maxPool1.printPoolMaps();
               conv2.printActivationMaps();
               maxPool2.printPoolMaps();
               flat.printAct();
           }
           out.printPrediction();
           errorCount += out.reportPredictionError();
       }
       System.out.println(out.getCountCorrect());
       return errorCount;
   }

    public int tuneCNN( Vector<Vector<Double>> tuneFeatureVectors) {

        int errorCount = 0;
    	out.resetCountCorrect();

    	
        for (int tuningIpNum = 0; tuningIpNum < tuneFeatureVectors.size(); tuningIpNum++) {

            Vector<Double> tuneFeatureVector = tuneFeatureVectors.get(tuningIpNum);

            conv1.train(tuneFeatureVector);
            maxPool1.train(conv1);
            conv2.train(maxPool1);
            maxPool2.train(conv2);
            flat.train(maxPool2);
            out.train(flat);

            if (debugCNN) {

                conv1.printActivationMaps();
                maxPool1.printPoolMaps();
                conv2.printActivationMaps();
                maxPool2.printPoolMaps();
                flat.printAct();
            }
            out.printPrediction();
            errorCount += out.reportPredictionError();
        }
        System.out.println(out.getCountCorrect());
        return errorCount;
    }

    public int testCNN( Vector<Vector<Double>> testFeatureVectors) {

 	   	out.resetCountCorrect();
 	   	out.zeroConfusionMatrix();

        int errorCount = 0;
 	   
        for (int testIpNum = 0; testIpNum < testFeatureVectors.size(); testIpNum++) {

            Vector<Double> testFeatureVector = testFeatureVectors.get(testIpNum);

            conv1.train(testFeatureVector);
            maxPool1.train(conv1);
            conv2.train(maxPool1);
            maxPool2.train(conv2);
            flat.train(maxPool2);
            out.train(flat);

            if (debugCNN) {

                conv1.printActivationMaps();
                maxPool1.printPoolMaps();
                conv2.printActivationMaps();
                maxPool2.printPoolMaps();
                flat.printAct();
            }
            out.printPrediction();
            errorCount += out.reportPredictionError();
        }
        System.out.println(out.getCountCorrect());
        out.printConfusion();
        return errorCount;
    }
   
}