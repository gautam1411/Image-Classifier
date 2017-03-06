/***
 *
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
    private FlatLayer flat;
    private OutputLayer out;


   public ConvNet( Vector<Vector<Double>> inputFeatureVectors , int hyperparameters, boolean debugSwitch){

       debugCNN     = debugSwitch;

       conv1        = new Convolution(inputFeatureVectors, hyperparameters, debugCNN);    // Conv-1
       maxPool1     = new Pooling(conv1, debugCNN);                                       // Pool-1
       conv2        = new Convolution(maxPool1, hyperparameters,  debugCNN);              // Conv-2
       maxPool2     = new Pooling(conv2, debugCNN);                                       // Pool-2
       flat         = new FlatLayer(maxPool2, debugCNN);                                  // Flat Fully Connected Layer
       out          = new OutputLayer(flat, hyperparameters, debugCNN);                   // Output Layer

   }


   public void trainCNN( Vector<Vector<Double>> trainFeatureVectors) {

       for (int trainingIpNum = 0; trainingIpNum < trainFeatureVectors.size(); trainingIpNum++) {

           Vector<Double> trainFeatureVector = trainFeatureVectors.get(trainingIpNum);

           conv1.train(trainFeatureVector);
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
       }
   }

    public void tuneCNN( Vector<Vector<Double>> tuneFeatureVectors) {

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
        }
    }

    public void testCNN( Vector<Vector<Double>> testFeatureVectors) {

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
        }
    }

}