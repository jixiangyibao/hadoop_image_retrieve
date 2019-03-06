/**
 * Created by 刘咏 on 2018/2/12.
 */
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import javax.imageio.ImageIO;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
public class FeatExtrLOMO extends Mapper<Text, BytesWritable, Text, Text>{
    public void map(Text k, BytesWritable v, Context contex) throws IOException, InterruptedException {
        byte[] data = new byte[v.getLength()];
        System.arraycopy(v.getBytes(), 0, data, 0, v.getLength());
        InputStream is = new ByteArrayInputStream(data);
        BufferedImage img = ImageIO.read(is);
        System.out.println(k);
        int[] R={3,5};
        int[] colorBins = {8, 8, 8};
        int numScales =3;
        int blockSize=10;
        int blockStep=5;
        ArrayList<Double> last_descriptor_list=new ArrayList();
        double[] lenArray=new double[R.length+1];
        int[] sizeArray=new int[R.length+1];
        int size=0;

        ArrayList<Double>descriptor1= PyramidMaxJointHist(img,numScales,blockSize,blockStep,colorBins);
        size+=descriptor1.size();
        //转化为数组
        double sum=0;
        for(int i=0;i<descriptor1.size();i++){
            sum+=descriptor1.get(i)*descriptor1.get(i);
            last_descriptor_list.add(descriptor1.get(i));
        }

        double len=Math.sqrt(sum);
        lenArray[0]=len;
        sizeArray[0]=descriptor1.size();

        int numPoints = 4;
        double tau=0.3;


        for(int i=0;i<R.length;i++) {
       // for(int i=0;i<1;i++) {
            ArrayList<Double> descriptor_tmp=PyramidMaxSILTPHist(img, numScales, blockSize, blockStep, tau, R[i], numPoints);
            sum=0;
            size+=descriptor_tmp.size();
            for(int j=0;j<descriptor_tmp.size();j++) {
                sum=sum+descriptor_tmp.get(j)*descriptor_tmp.get(j);
                last_descriptor_list.add(descriptor_tmp.get(j));
            }
            len=Math.sqrt(sum);
            lenArray[1+i]=len;
            sizeArray[1+i]=descriptor_tmp.size()+sizeArray[i];

         }

        //normc
        int j=0;
        double[] last_descriptor=new double[size];
        for(int i=0;i<R.length+1;i++){
            for(;j<sizeArray[i];j++){
                last_descriptor[j]=last_descriptor_list.get(j)/lenArray[i];
            }
            j=sizeArray[i];
        }
      //  int index=k.toString().indexOf("cam");
        int index=0;
        String s=k.toString().substring(index);

        Text value=new Text();
        String tmp="";
        for(int i=0;i<size;i++){
            tmp=tmp+last_descriptor[i]+",";
        }
        value.set(tmp);
        Text key=new Text(s);
        contex.write(key,value);

    }
    public  ArrayList<Double> PyramidMaxJointHist(BufferedImage oriImgs, int numScales, int blockSize, int blockStep,int[] colorBins){
        int height = oriImgs.getHeight();
        int width = oriImgs.getWidth();
        int totalBins = 512;
        int  minRow = 1;
        int  minCol = 1;
        double[][][] imgHSB =new double[height][width][3];
        for (int i = 0; i < width ; i++) {
            for (int j = 0; j < height; j++) {
                int rgb = oriImgs.getRGB(i, j);
                int r = (rgb >> 16) & 0xff;
                int g = (rgb >> 8) & 0xff;
                int b = (rgb) & 0xff;
                float[] hsb=new float[3];
                //convert RGB to HSB
                Color.RGBtoHSB(r, g, b, hsb);
                imgHSB[j][i][0]=hsb[0];
                imgHSB[j][i][1]=hsb[1];
                imgHSB[j][i][2]=hsb[2];
                //HSV量化
                imgHSB[j][i][0]=Math.min(Math.floor(imgHSB[j][i][0]*colorBins[0]),colorBins[0]-1);
                imgHSB[j][i][1]=Math.min(Math.floor(imgHSB[j][i][1]*colorBins[1]),colorBins[1]-1);
                imgHSB[j][i][2]=Math.min(Math.floor(imgHSB[j][i][2]*colorBins[2]),colorBins[2]-1);

            }
        }
        // Scan multi-scale blocks and compute histograms
        // int[] descriptors=new int[totalBins*rowLength*numScales];
        ArrayList<Double> descriptor1 = new ArrayList();

        for(int ii=0 ; ii< numScales; ii++){

            int maxRow = height - blockSize + 1;
            int maxCol = width - blockSize + 1;
            int colLength=(maxCol-minCol)/blockStep+1;
            int rowLength=(maxRow-minRow)/blockStep+1;
            int numBlocks=rowLength*colLength;
            double [] patterns=new double[height*width];

            int ind=0;
            //编码
            for (int i = 0; i < width  ; i++) {
                for (int j = 0; j < height; j++) {
                    patterns[ind++] = imgHSB[j][i][2] * colorBins[1] * colorBins[0] +
                            imgHSB[j][i][1] * colorBins[0] +
                            imgHSB[j][i][0];
                }
            }

            //复制行编号，列编号
            int[] cols= new int[numBlocks];
            int[] rows =new int[numBlocks];
            for(int i=0;i<colLength;i++){
                for(int j=0;j<rowLength;j++){
                    cols[i*rowLength+j]=minCol+blockStep*i;
                    rows[i*rowLength+j]=minRow+blockStep*j;
                }
            }


            if (numBlocks == 0){
                break;
            }

             //与左上角位置的偏移量
            int sqBlocksize=blockSize*blockSize;
            int[] offset=new int[sqBlocksize];
            for(int i=0;i<blockSize;i++){
                for (int j=0;j<blockSize;j++){
                    offset[i*blockSize+j]=i*height+j;
                }
            }

            //计算每个子窗口的起始位置
            int[] index_tmp=new int[numBlocks];
            for(int i=0;i<numBlocks;++i){
                index_tmp[i]=rows[i]+height*(cols[i]-1);
            }

            //计算每个子窗口的坐标
            int[] index= new int[sqBlocksize*numBlocks];
            for(int i=0;i<numBlocks ;i++){
                for(int j=0;j<sqBlocksize;j++){
                    index[i*sqBlocksize+j]=(index_tmp[i]+offset[j])-1;
                }
            }


            //计算每一个子窗口的联合HSV值
            double[][] patches=new double[sqBlocksize][numBlocks];
            ind=0;
            for(int i=0;i<numBlocks;i++){
                for(int j=0;j<sqBlocksize;j++){
                    patches[j][i]=patterns[index[ind++]];
                }
            }


            //HISTOGRAM

            int[][] hist_feature = new int[totalBins][numBlocks];
            for(int i=0;i<numBlocks;i++){
                for(int j=0;j<sqBlocksize;j++){
                    hist_feature[(int)Math.round(patches[j][i])][i]++;
                }
            }


            //reshape hist_feature
            int[][][] reshape_hist_feature = new int[totalBins][rowLength][colLength];
            for(int i=0;i<totalBins;i++){
                for(int j=0;j<colLength;j++){
                    for(int z=0;z<rowLength ;z++){
                        reshape_hist_feature[i][z][j]=hist_feature[i][z+j*rowLength];
                    }
                }
            }


            int[][] max_hist_feature=new int[totalBins][rowLength];
            for(int i=0;i<totalBins;i++){
                for(int j=0;j<rowLength;j++){
                    int maxValue=Integer.MIN_VALUE;
                    for(int z=0;z<colLength;z++){
                        if(reshape_hist_feature[i][j][z]>maxValue){
                            maxValue=reshape_hist_feature[i][j][z];
                        }
                    }
                    max_hist_feature[i][j]=maxValue;
                }
            }

            //展开

            int[] fea=new int[totalBins*rowLength];
            ind=0;
            for(int i=0;i<rowLength ;i++){
                for(int j=0;j<totalBins;j++){
                    fea[ind]=max_hist_feature[j][i];
                    descriptor1.add(Math.log(fea[ind]+1));
                    ind++;
                }
            }


            //ColorPooling
            if(ii < numScales-1){
                int out_height=height;
                int out_width=width;
                if (out_height%2==1){
                    out_height=out_height-1;
                }
                if (out_width%2==1){
                    out_width=out_width-1;
                }
                out_height=out_height/2;
                out_width=out_width/2;

                double[] imgHSB_vector =new double[height*width*3];
                ind=0;
                for(int z=0;z<3;z++){
                    for(int i=0;i<width;i++){
                        for(int j=0;j<height;j++){
                            imgHSB_vector[ind++]=imgHSB[j][i][z];
                            //  System.out.println(imgHSB_vector[ind-1]);
                        }
                    }
                }
                ind=0;
                double[][][][][] reshape_imgHSB=new double[2][out_height][2][out_width][3];

                for(int channel_ind=0;channel_ind<3;channel_ind++){
                    for(int width_ind=0;width_ind<out_width;width_ind++){
                        for(int half_ind1=0;half_ind1<2;half_ind1++){
                            for(int heig_ind=0;heig_ind<out_height;heig_ind++){
                                for(int half_ind2=0;half_ind2<2;half_ind2++){
                                    reshape_imgHSB[half_ind2][heig_ind][half_ind1][width_ind][channel_ind]=imgHSB_vector[ind++];
                                }
                            }
                        }
                    }
                }

                double[] permute_imgHSB_vector =new double[height*width*3];
                ind=0;
                for(int ind1=0;ind1<2;ind1++) {
                    for (int ind2 = 0; ind2 < 2; ind2++) {
                        for (int ind3 = 0; ind3 < 3; ind3++){
                            for (int ind4 = 0; ind4 < out_width; ind4++){
                                for (int ind5 = 0; ind5 < out_height; ind5++) {
                                    permute_imgHSB_vector[ind++] = reshape_imgHSB[ind2][ind5][ind1][ind4][ind3];
                                }
                            }
                        }
                    }
                }


                double[][][][] permute_imgHSB=new double[out_height][out_width][3][4];
                ind=0;
                for(int ind4=0;ind4<4;ind4++){
                    for(int ind3=0;ind3<3;ind3++){
                        for(int ind2=0;ind2<out_width;ind2++){
                            for(int ind1=0;ind1<out_height;ind1++){
                                permute_imgHSB[ind1][ind2][ind3][ind4]=permute_imgHSB_vector[ind++];
                            }
                        }

                    }
                }

                for(int ind1=0;ind1<out_height;ind1++){
                    for(int ind2=0;ind2<out_width;ind2++){
                        for(int ind3=0;ind3<3;ind3++){
                            double tmp=0;
                            for(int ind4=0;ind4<4;ind4++){
                                tmp+=permute_imgHSB[ind1][ind2][ind3][ind4];
                            }
                            tmp=Math.floor(tmp/4);
                            imgHSB[ind1][ind2][ind3]=tmp;
                        }
                    }

                }
                height=out_height;
                width=out_width;
            }

        }
        return descriptor1;
    }
    public  ArrayList<Double> PyramidMaxSILTPHist(BufferedImage oriImgs, int numScales, int blockSize, int blockStep,double tau,int R,int numPoints ){
        int totalBins=(int)Math.pow(3,numPoints);
        int height = oriImgs.getHeight();
        int width = oriImgs.getWidth();
        //灰度值

        double[][] imgGray =new double[height][width];
        for (int i = 0; i < width ; i++) {
            for (int j = 0; j < height; j++) {
                int rgb = oriImgs.getRGB(i, j);
                double r = (rgb >> 16) & 0xff;
                double g = (rgb >> 8) & 0xff;
                double b = (rgb) & 0xff;
                imgGray[j][i]=(0.299 * r + 0.587 * g + 0.114 * b)/255;
            }
        }

        int  minRow = 1;
        int  minCol = 1;
        ArrayList<Double> descriptor = new ArrayList();

        for(int ii=0 ; ii< numScales; ii++) {
            if(width<R*2+1){
                System.out.println("skip scale R="+R+",width="+width+".");
                continue;
            }
            int[][] J= SILTP(imgGray, tau, R, numPoints,height,width);
            int[] patterns=new int[height*width];
            int ind=0;
            for(int j=0;j<width;j++){
                for(int i=0;i<height;i++){
                    patterns[ind++]=J[i][j];
                }
            }
            int maxRow = height - blockSize + 1;
            int maxCol = width - blockSize + 1;
            int colLength=(maxCol-minCol)/blockStep+1;
            int rowLength=(maxRow-minRow)/blockStep+1;
            int numBlocks=rowLength*colLength;
            int[] cols= new int[numBlocks];
            int[] rows =new int[numBlocks];
            for(int i=0;i<colLength;i++){
                for(int j=0;j<rowLength;j++){
                    cols[i*rowLength+j]=minCol+blockStep*i;
                    rows[i*rowLength+j]=minRow+blockStep*j;
                    //System.out.println(cols[i*rowLength+j]);
                }
            }
            if (numBlocks == 0){
                break;
            }
            int sqBlocksize=blockSize*blockSize;
            int[] offset=new int[sqBlocksize];
            for(int i=0;i<blockSize;i++){
                for (int j=0;j<blockSize;j++){
                    offset[i*blockSize+j]=i*height+j;
                    // System.out.println(offset[i*blockSize+j]);
                }
            }
            int[] index_tmp=new int[numBlocks];
            for(int i=0;i<numBlocks;++i){
                index_tmp[i]=rows[i]+height*(cols[i]-1);
                // System.out.println(index_tmp[i]);
            }
            int[] index= new int[sqBlocksize*numBlocks];
            for(int i=0;i<numBlocks ;i++){
                for(int j=0;j<sqBlocksize;j++){
                    index[i*sqBlocksize+j]=(index_tmp[i]+offset[j])-1;
                    // System.out.println(last_index[i*sqBlocksize+j]);
                }
            }
            double[][] patches=new double[sqBlocksize][numBlocks];

            ind=0;
            for(int i=0;i<numBlocks;i++){
                for(int j=0;j<sqBlocksize;j++){
                    patches[j][i]=patterns[index[ind++]];
                }
            }



            //HISTOGRAM
            int[][] hist_feature = new int[totalBins][numBlocks];
            for(int i=0;i<numBlocks;i++){
                for(int j=0;j<sqBlocksize;j++){
                    hist_feature[(int)Math.round(patches[j][i])][i]++;
                }
            }


            //reshape hist_feature
            int[][][] reshape_hist_feature = new int[totalBins][rowLength][colLength];
            for(int i=0;i<totalBins;i++){
                for(int j=0;j<colLength;j++){
                    for(int z=0;z<rowLength ;z++){
                        reshape_hist_feature[i][z][j]=hist_feature[i][z+j*rowLength];
                    }
                }
            }
            int[][] max_hist_feature=new int[totalBins][rowLength];
            for(int i=0;i<totalBins;i++){
                for(int j=0;j<rowLength;j++){
                    int maxValue=Integer.MIN_VALUE;
                    for(int z=0;z<colLength;z++){
                        if(reshape_hist_feature[i][j][z]>maxValue){
                            maxValue=reshape_hist_feature[i][j][z];
                        }
                    }
                    max_hist_feature[i][j]=maxValue;
                }
            }

            int[] fea=new int[totalBins*rowLength];
            ind=0;
            for(int i=0;i<rowLength ;i++){
                for(int j=0;j<totalBins;j++){
                    fea[ind]=max_hist_feature[j][i];
                    //   descriptors[ind+ii*totalBins*rowLength]=fea[ind];
                    descriptor.add(Math.log(fea[ind]+1));
                    ind++;
                }
            }

            //Pooling
            if(ii < numScales-1) {
                int out_height=height;
                int out_width=width;
                if (out_height%2==1){
                    out_height=out_height-1;
                }
                if (out_width%2==1){
                    out_width=out_width-1;
                }
                out_height=out_height/2;
                out_width=out_width/2;

                double[] imgGray_vector =new double[height*width];
                ind=0;
                for(int i=0;i<width;i++){
                    for(int j=0;j<height;j++){
                        imgGray_vector[ind++]=imgGray[j][i];
                        //  System.out.println(imgHSB_vector[ind-1]);
                    }
                }

                ind=0;
                double[][][][] reshape_imgGray=new double[2][out_height][2][out_width];


                    for(int width_ind=0;width_ind<out_width;width_ind++){
                        for(int half_ind1=0;half_ind1<2;half_ind1++){
                            for(int heig_ind=0;heig_ind<out_height;heig_ind++){
                                for(int half_ind2=0;half_ind2<2;half_ind2++){
                                    reshape_imgGray[half_ind2][heig_ind][half_ind1][width_ind]=imgGray_vector[ind++];
                                }
                            }
                        }
                    }
                double[] permute_imgGray_vector =new double[height*width];
                ind=0;
                for(int ind1=0;ind1<2;ind1++) {
                    for (int ind2 = 0; ind2 < 2; ind2++) {
                            for (int ind3 = 0; ind3 < out_width; ind3++){
                                for (int ind4 = 0; ind4 < out_height; ind4++) {
                                    permute_imgGray_vector[ind++] = reshape_imgGray[ind2][ind4][ind1][ind3];
                                }

                        }
                    }
                }

                double[][][] permute_imgGray=new double[out_height][out_width][4];
                ind=0;
                for(int ind3=0;ind3<4;ind3++){
                        for(int ind2=0;ind2<out_width;ind2++){
                            for(int ind1=0;ind1<out_height;ind1++){
                                permute_imgGray[ind1][ind2][ind3]=permute_imgGray_vector[ind++];
                            }
                    }
                }

                for(int ind1=0;ind1<out_height;ind1++){
                    for(int ind2=0;ind2<out_width;ind2++){
                            double tmp=0;
                            for(int ind3=0;ind3<4;ind3++){
                                tmp+=permute_imgGray[ind1][ind2][ind3];
                            }
                            tmp=tmp/4;
                            imgGray[ind1][ind2]=tmp;
                        }

                }
                height=out_height;
                width=out_width;
            }

        }

        return descriptor;
    }
    public  int[][] SILTP(double[][] oriImg, double tau,int  R,int numPoints,int height,int width){
        int encoder=0;
        if(height<2*R+1 || width <2*R+1){
            System.out.println("Too small image or too large R");
        }

        //put the image in a larger container

        double[][] I0=new double[height+2*R][width+2*R];
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                I0[i+R][j+R]=oriImg[i][j];
            }
        }

        //replicate border image pixels to the outer area
        for(int i=0;i<R;i++){
            for(int j=R;j<width+R;j++){
                I0[i][j]=I0[R][j];
            }
        }
        for(int i=height+R;i<height+2*R;i++){
            for(int j=R;j<width+R;j++) {
                I0[i][j]=I0[height+R-1][j];
            }
        }
       for(int j=0;j<R;j++){
            for(int i=0;i<height+2*R;i++){
                I0[i][j]=I0[i][R+1];
            }
       }

       for(int j=width+R;j<width+2*R;j++){
           for(int i=0;i<height+2*R;i++){
               I0[i][j]=I0[i][width+R-1];
           }
       }

       //copy image in specified directions

        double[][] I1=new double[height][width];
        for(int i=R;i<height+R;i++){
            for(int j=2*R;j<width+2*R;j++){
                I1[i-R][j-2*R]=I0[i][j];
            }
        }

        double[][] I3=new double[height][width];
        for(int i=0;i<height;i++){
            for(int j=R;j<width+R;j++){
                I3[i][j-R]=I0[i][j];
            }
        }

        double[][] I5=new double[height][width];
        for(int i=R;i<height+R;i++){
            for(int j=0;j<width;j++){
                I5[i-R][j]=I0[i][j];
            }
        }

        double[][] I7=new double[height][width];
        for(int i=R*2;i<height+2*R;i++){
            for(int j=R;j<width+R;j++){
                I7[i-2*R][j-R]=I0[i][j];
            }
        }

        //compute the upper and lower range
        double[][] L=new double[height][width];
        double[][] U=new double[height][width];
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                L[i][j]=(1-tau)*oriImg[i][j];
                U[i][j]=(1+tau)*oriImg[i][j];
            }
        }
        //compute the scale invariant local ternary patterns
        int[][] J=new int[height][width];
        int tmp=0;
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                tmp=0;
                if(I1[i][j]<L[i][j]){
                    tmp+=1;
                }
                if(I1[i][j]>U[i][j]){
                    tmp+=2;
                }
                if(I3[i][j]<L[i][j]){
                    tmp+=3;
                }
                if(I3[i][j]>U[i][j]){
                    tmp+=6;
                }
                if(I5[i][j]<L[i][j]){
                    tmp+=9;
                }
                if(I5[i][j]>U[i][j]){
                    tmp+=18;
                }
                if(I7[i][j]<L[i][j]){
                    tmp+=27;
                }
                if(I7[i][j]>U[i][j]){
                    tmp+=54;
                }
               J[i][j]=tmp;
            }
        }
        return J;
    }
}

