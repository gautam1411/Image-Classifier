/**
*
 *
 *
 *  Class PoolPlate describes plates in a pooling layer and related methods
 */


public class PoolMap{


    private Double label;

    private Double [][] inputFeature;
    private Double [][] outputMap;
    private Double [][] errors;

    //private int poolRatio;
    private boolean debugPoolMap = false;

    private int plateSize;
    private int outVol;



    public PoolMap(int plateSize, int outVol, boolean debugSwitch){

        debugPoolMap = debugSwitch;

        if(debugSwitch)
            System.out.println("Input plate size: "+plateSize + " Output plate size: " + outVol);

        if( plateSize <= 0 || outVol <= 0 ){

            System.out.println("<PoolMap> : Invalid parameter");
        }else {
            inputFeature = new Double[plateSize][plateSize];
            outputMap = new Double[outVol][outVol];
            errors = new Double[outVol][outVol];
            debugPoolMap = debugSwitch;

            this.plateSize = plateSize;
            this.outVol = outVol;

            //poolRatio = plateSize/outVol ;
        }

    }


    public void maxPool(){

        int poolRatio = 2; // Default value

        if( outputMap.length != 0) {
            poolRatio = inputFeature.length / outputMap.length;
        }else{
            System.out.println("<PoolMap> : Invalid parameter");
        }

        //if(debugPoolMap)
         //   System.out.println("<PoolMap>: poolRatio : " +poolRatio );

        for(int i = 0; i< outputMap.length; i++)
            for(int j =0 ; j < outputMap[0].length;  j++){

                outputMap[i][j] = maxPoolHelper(i,j);
            }
    }

    public Double maxPoolHelper(int row, int col){

        int poolRatio = 2; // Default value

        if(outputMap.length != 0) {

           poolRatio =  inputFeature.length / outputMap.length;
        }else{
            System.out.println(" <PoolMap> : Invalid parameter");
        }

        Double max = Double.MIN_VALUE;

        for(int i = poolRatio* row; i < (row+1)*poolRatio ; i++) {
            for (int j = poolRatio * col; j < (col + 1) * poolRatio; j++) {

                if (inputFeature[i][j] > max)
                    max = inputFeature[i][j];
            }
        }

        if(debugPoolMap)
            System.out.println("<PoolMap>: max value : " +max );

        return max;
    }


    public void avgPool(){

        int poolRatio = 2; // Default value

        if(outputMap.length != 0) {
            poolRatio = inputFeature.length / outputMap.length;
        }else{
            System.out.println("<PoolMap> : Invalid parameter");
        }

        if(debugPoolMap)
            System.out.println("<PoolMap>: poolRatio : " +poolRatio );

        for(int i = 0; i< outputMap.length; i++)
            for(int j =0 ; j < outputMap[0].length;  j++){

                outputMap[i][j] = avgPoolHelper(i,j);
            }
    }

    public Double avgPoolHelper(int row, int col){

        int poolRatio = 2; // Set default value

        if( outputMap.length != 0) {
            poolRatio = inputFeature.length / outputMap.length;
        }else{

            System.out.println("Invalid parameter ");
        }

        Double sum = 0.0;

        for(int i = poolRatio* row; i < (row+1)*poolRatio ; i++) {
            for (int j = poolRatio * col; j < (col + 1) * poolRatio; j++) {

                    sum += inputFeature[i][j];
            }
        }

        Double avg = sum/(poolRatio*poolRatio);

        if(debugPoolMap)
            System.out.println("<PoolMap>: avg value : " +avg );

        return avg;
    }

    public Double [][] getInputFeature(){

        return inputFeature;
    }

    public void computePoolMap(){

        // Maxpool have been found to work better. we can test avgpool also
        maxPool();
       // avgPool();

    }

    public void printPoolMap(){


    }

    public int getPlateSize(){

        return plateSize;
    }
    public int getOutVol(){

        return outVol;
    }

    public Double [][] getOutput (){


        return  outputMap;
    }

    public void backpropagate(){


    }

    public Double[][] getErrors(){

        return errors;
    }

}