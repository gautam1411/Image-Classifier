/**
*
 *
 *
 * This class defines Output Layer
 *
 */

import java.util.Random;

public class OutputLayer{


    private Double [] input;
    private Double [] output;
    private Double [] expected;
    private Double [][] weights;

    private boolean debugOutputLayer = false;
    private int countClasses;

    private Double label;


    public OutputLayer(){

        if(debugOutputLayer)
        System.out.println("<OutputLayer> : OutputLayer Default Constructor");
        output = new Double[countClasses];
        expected = new Double[countClasses];
    }

    public OutputLayer(FlatLayer flat, int hyperparameters, boolean debugSwitch){

        debugOutputLayer = debugSwitch;
        countClasses = (hyperparameters >> 24)& (0XFF);

        if(debugOutputLayer) {
            System.out.println("<OutputLayer> : OutputLayer Constructor previous layer as FlatLayer ");
            System.out.println("<OutputLayer> : Number of classes : "+countClasses);

        }

        int countInputs = flat.getCountInputs();

        if(debugOutputLayer)
        System.out.println(" input array size "+countInputs);

        input = new Double [ countInputs ];
        output = new Double[countClasses];
        expected = new Double[countClasses];
        weights = new Double [countClasses][countInputs];
        initWeights();
    }

    public void initWeights(){

        Random rand = new Random();
        Random randSign = new Random();

        for(int i = 0; i < weights.length; i++){

            for(int j = 0; j < weights[0].length; j++){

                int sign = randSign.nextInt() % 2;
                weights[i][j] = rand.nextDouble();

                if( sign == 0)
                    weights[i][j] *= -1;

            }

        }

    }


    public void train(FlatLayer flat){


        for (int i = 0; i< 6 ; i++){

            Double sum = 0.0;

            for( int j = 0; j < weights[0].length; j++) {

                sum += weights[i][j] /* * input[j] */ ;
            }
            output[i] = sum;

            System.out.println( " Prediction : " +i + " >  " +sum);

        }

    }

}