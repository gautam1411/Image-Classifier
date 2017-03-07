/**********************
 *
 *
 *
 *  Convolution class holds all feature(activation) maps in the convolution layer and related methods
 */

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import java.util.Arrays;

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

        if(kernel_size <=0 || countFeatureMaps <=0){

            System.out.println("Inavlid parameter passes to Convolution layer constructor");
        }

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

        this.label = featureVector.get(featureVector.size()-1);

        if(debugconv)
            System.out.println(" Image Label : " + debugconv);

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

        this.label = pool.getLabel();

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

    public Double getLabel(){

        return this.label;
    }


/*
    public void backpropagate(Pooling pool){


        Double learning_rate = 0.01;

        ArrayList<PoolMap> pmaps = pool.get_P_maps();

        // Copy error vector from Pooling layer

       for ( int i = 0; i < pmaps.size(); i ++){

           PoolMap pm =pmaps.get(i);
           FeatureMap fm = feature_maps.get(i);

           Double [][] err = pm.getErrors();

           Double [][] error = fm.getErrors();

           for(int j = 0; j < err.length; j++){

               System.arraycopy(err[j],0, error[j],0,err[j].length);
           }

       }

        // For each plate ( feature/ activation map) do the following
        // <1> For each entry in error matrix
        // <2> Find winner value corresponsing to output featuremap

        for( int i = 0 ; i < feature_maps.size(); i++){


            FeatureMap fm = feature_maps.get(i);

            Double [][] err = fm.getErrors();
            Double [][] out = fm.getFeatureMap();
            Double [][] ker = fm.getKernel();
            Double [][] inp = fm.getInputMap();

            System.out.println(" Error vector dimension :"+err.length+ "  "+err[0].length);

            for(int j=0; j < err.length; j++){

                for(int k = 0; k < err[0].length; k++){

                    // find indices of winner value in output  matrix
                    int downsample_ratio = input_size / outputVol ;

                    int ind1 =  -1;
                    int ind2 =  -1;
                    Double max_val = Double.MIN_VALUE;

                    for( int l = downsample_ratio * j;  l < (downsample_ratio *(j+1)) ; l++ ){

                        for( int m = downsample_ratio * k;  m < (downsample_ratio *(k+1)) ; m++ ){

                            if(out[l][m] > max_val){
                                max_val = out[l][m];
                                ind1 = l;
                                ind2 = m;
                            }
                        }

                    }

                    if(true){
                        System.out.println(" Max value: "+max_val + " , Ind1: "+ind1+"  ,Ind2  : "+ind2);
                        System.out.println("<Convolution.java> : j,k  > "+j+"  "+k);
                    }

                    if(ind1 == -1 || ind2 == -1){

                        if(true)
                        System.out.println("<Convolution.java> : Failed to find indices of winner value");
                    }else{

                        // Remember we have calculated activation of neuron as
                        // featureMap[i][j] = RELU(activation (i,j) );
                        // we should compute gradient and update kernel

                        for(int p=0; p < kernel_size; p++) {
                            for (int q = 0; q < kernel_size; q++) {

                                System.out.println(" "+p+" "+q+" "+j+" "+k+" "+ind1+" "+ind2);
                                ker[p][q] += err[j][k]* learning_rate * inp[p+ind1][q+ind2];
                            }
                        }

                    }

                }
            }

        }

    }
    */

    public void backpropagate(Pooling pool){


        Double learning_rate = 0.01;

        ArrayList<PoolMap> pmaps = pool.get_P_maps();


        for ( int i = 0; i < pmaps.size(); i ++){

            PoolMap pm =pmaps.get(i);
            Double [][] err = pm.getErrors();

            FeatureMap fm = feature_maps.get(i);
            Double [][] error = fm.getErrors();
            Double[][] out = fm.getFeatureMap();
            //Double[][] ker = fm.getKernel();
            //Double[][] inp = fm.getInputMap();

            System.out.println(" Error vector dimension :" + err.length + "  " + err[0].length);

            for( int x = 0 ;x<error.length ; x++ )
                Arrays.fill(error[x], 0.0);


            for(int j = 0; j < err.length; j++){

                for(int k = 0; k < err[0].length ; k++){


                    int downsample_ratio = input_size / outputVol;

                    int ind1 = -1;
                    int ind2 = -1;
                    Double max_val = Double.MIN_VALUE;

                    for (int l = downsample_ratio * j; l < (downsample_ratio * (j + 1)); l++) {

                        for (int m = downsample_ratio * k; m < (downsample_ratio * (k + 1)); m++) {

                            if (out[l][m] > max_val) {
                                max_val = out[l][m];
                                ind1 = l;
                                ind2 = m;
                            }
                        }

                    }

                    if (true) {
                        System.out.println(" Max value: " + max_val + " , Ind1: " + ind1 + "  ,Ind2  : " + ind2);
                        System.out.println("<Convolution.java> : j,k  > " + j + "  " + k);
                    }

                    if (ind1 == -1 || ind2 == -1) {

                        if (true)
                            System.out.println("<Convolution.java> : Failed to find indices of winner value");
                    }else{

                        error[ind1][ind2] = err[j][k];

                    }

                }
            }

        }

        // For each plate ( feature/ activation map) do the following
        // <1> For each entry in error matrix

        for( int i = 0 ; i < feature_maps.size(); i++){


            FeatureMap fm = feature_maps.get(i);

            Double [][] err = fm.getErrors();
            Double [][] out = fm.getFeatureMap();
            Double [][] ker = fm.getKernel();
            Double [][] inp = fm.getInputMap();

            if(debugconv)
            System.out.println(" Error vector dimension :"+err.length+ "  "+err[0].length);

            for(int j=0; j < err.length; j++){

                for(int k = 0; k < err[0].length; k++){
                    // Remember we have calculated activation of neuron as
                    // featureMap[i][j] = RELU(activation (i,j) );
                    // we should compute gradient and update kernel

                    if(err[j][k] != 0.0) {

                        for (int p = 0; p < kernel_size; p++) {

                            for (int q = 0; q < kernel_size; q++) {

                                if(debugconv)
                                System.out.println(" " + p + " " + q + " " + j + " " + k + " ");

                                ker[p][q] += err[j][k] * learning_rate * inp[p + j][q + k];
                            }
                        }
                    }
                }
            }

        }

    }

    public int getKernelSize(){

        return kernel_size;

    }

}