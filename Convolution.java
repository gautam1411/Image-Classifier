/**********************
 *
 *
 *
 *  Convolution class holds all feature(activation) maps in the convolution layer and related methods
 */

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class Convolution{

    // Boolean flag to switch debug at Convolution layer
    private boolean debugconv = true;

    // List of all feature maps or activation maps in the convolution layer
    private ArrayList<FeatureMap> feature_maps;
    // Size of filter or kernel
    private int kernel_size;
    // Stride
    private int stride;
    // Padding
    private int padding;
    // Number of feature maps
    public int countFeatureMaps;
    // Image size
    private int input_size;
    // Output size
    public int outputVol;

    private Double label;


    public Convolution(){

        System.out.println("  Conv Layer Default constructor ");

        this.feature_maps = new ArrayList<FeatureMap>();
    }


    public Convolution( Vector<Vector<Double>> inputFeatureVectors, int hyperparameters, boolean debugSwitch){

        debugconv = debugSwitch;
        setHyperParameters( hyperparameters);


        this.feature_maps = new ArrayList<FeatureMap> ();

        input_size = (int ) Math.round( Math.sqrt(inputFeatureVectors.get(0).size()-1) );

        if(debugconv){
            System.out.println(" Conv Layer Constructor ");
            System.out.println(" Trace <ConvLayer>:  Number of Input Vectors : "+ inputFeatureVectors.size())  ;
            System.out.println(" Trace <ConvLayer>:  Input Vector size       :  "+ inputFeatureVectors.get(0).size());
            // 2D Image is encoded into 1-D feature vector + label as last item
            System.out.println(" Trace <ConvLayer>:  Input Vector Dimension: " + input_size);
            System.out.println("Pending <ConvLayer>: Testing with kernel size 5 now ");
        }

        outputVol = outputVolume();
        if(debugconv){
            System.out.println("Input: " + input_size + " Kernel: " + kernel_size + " Output: " + outputVol);
            System.out.println("Pending <ConvLayer>: Testing code with padding : " + padding);
            System.out.println("Pending <ConvLayer>: Testing with stride : " + stride);
        }
        // Add countFeatureMaps FeatureMaps at Convolution Layer
        for(int i =0; i< countFeatureMaps; i++){

            FeatureMap featureMap = new FeatureMap(input_size, kernel_size,outputVol, debugSwitch);
            addFeatureMap(featureMap);
            featureMap.initKernel();
        }
    }


    public void setHyperParameters( int hyperParameter){

        padding                 =  (hyperParameter >>28)  & (0xF);
        stride                  =  (hyperParameter >> 16) & (0xFF);
        kernel_size             =  (hyperParameter >>8)   & (0xFF);
        countFeatureMaps        =   hyperParameter        & (0xFF);

        if( debugconv)
        System.out.println(" "+padding+"   "+stride + "   "+ kernel_size+ "   "+ countFeatureMaps);

    }

    public Convolution( Pooling poolLayer, int hyperparameters, boolean debugSwitch){

        debugconv = debugSwitch;
        setHyperParameters( hyperparameters);


        if(debugconv)
        System.out.println(" Conv Layer Constructor ");

        this.feature_maps = new ArrayList<FeatureMap> ();


        input_size = poolLayer.outputVolume();



        if(debugconv){
            //System.out.println(" Trace <ConvLayer>:  Number of Input Vectors : "+ inputFeatureVectors.size())  ;
            //System.out.println(" Trace <ConvLayer>:  Input Vector size       :  "+ inputFeatureVectors.get(0).size());
            // 2D Image is encoded into 1-D feature vector + label as last item
            System.out.println(" Trace <ConvLayer>:  Input Vector Dimension: " + input_size);
            System.out.println("Pending <ConvLayer>: Testing with kernel size 5 now ");
        }

        outputVol = outputVolume();
        if(debugconv){
            System.out.println("Input: " + input_size + " Kernel: " + kernel_size + " Output: " + outputVol);
            System.out.println("Pending <ConvLayer>: Testing code with padding : " + padding);
            System.out.println("Pending <ConvLayer>: Testing with stride : " + stride);
        }
        // Add countFeatureMaps FeatureMaps at Convolution Layer
        for(int i =0; i< countFeatureMaps; i++){

            FeatureMap featureMap = new FeatureMap(input_size, kernel_size,outputVol,debugSwitch);
            addFeatureMap(featureMap);
            featureMap.initKernel();
        }
    }



    public void calcFeatureMaps( ){

        if(debugconv)
        System.out.println(" <ConvLayer> : calcFeatureMaps");

        for(FeatureMap feature_map: feature_maps)
            feature_map.computeFeatureMap(stride, padding);

    }

    // Get number of feature maps in the convolution layer
    public int countFeatureMaps(){

        return feature_maps.size();
    }

    // Add a feature map in the convolution layer
    public void addFeatureMap(FeatureMap featureMap){

        feature_maps.add(featureMap);
    }

    // Returns list of feature maps in the convolution layer
    public ArrayList<FeatureMap> get_fMaps(){

        return feature_maps;
    }


    public void readInputFeature(Vector <Double> featureVector){

        for (FeatureMap feature_map: feature_maps){

            feature_map.readFeatureVector(featureVector);


        }
    }



    public void readInputFeature(ArrayList<PoolMap> pMaps){


        for(int i = 0; i < pMaps.size(); i++){
            PoolMap pMap = pMaps.get(i);

            FeatureMap fMap = feature_maps.get(i);
            Double [][] inp = fMap.getInputMap();

            Double [][] input  = pMap.getOutput();


            for( int j = 0; j< input.length; j++)
                System.arraycopy(input[j],0, inp[j] , 0, input[j].length);

        }

    }


    public void printActivationMaps(){

        for (FeatureMap feature_map: feature_maps){

            feature_map.printActivationMap();

        }

    }

    public void train(  Vector<Double> inputFeatureVector){

            readInputFeature(inputFeatureVector);
            calcFeatureMaps();
    }

    public void train( Pooling pool){

        readInputFeature(pool.get_P_maps());
        calcFeatureMaps();


    }

    public int outputVolume(){

        // image should be properly padded if required for a particular kernel size
        int outVolume = ((input_size - kernel_size + 2 * padding)/ stride ) + 1;

        // Validates  if input feature size, kernel size , padding and stride values are okay
        // input feature should be padded if filter(kernel) size and stride are not tuned properly

        if( /*debugconv && */ (outVolume-1)*stride  !=  (input_size - kernel_size + 2 *padding) ){
           System.out.println("Error: <Convolution Layer>    <Hyperparameters Settings>");
        }
        return outVolume;
    }


}