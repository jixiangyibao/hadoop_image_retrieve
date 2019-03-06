import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import org.apache.commons.io.FileUtils;
import java.io.File;
import java.util.List;
import java.util.ArrayList;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.fs.FileSystem;
import java.net.URI;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.Path;
import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * Created by 刘咏 on 2018/2/21.
 */
public class knnMapper  extends Mapper<LongWritable, Text, Text,Text> {
    int featureSize=26960;
    public void map(LongWritable k, Text v, Context contex) throws IOException, InterruptedException {
        String[] tmp=v.toString().split("\t");
        String name=tmp[0];
        String[] fea=tmp[1].split(",");
        double[] feature=new double[featureSize];
        for(int i=0;i<fea.length;i++){
            feature[i]= Double.parseDouble(fea[i]);
        }
        double min=Double.MAX_VALUE;
        String minIndex="";

        String uri="hdfs://114.212.190.91:9000/user/2017st19/lab6/prid2011cambFea/part-r-00000";
        Configuration configuration = new Configuration();
        FileSystem fs=FileSystem.get(URI.create(uri), configuration);
        Path path = new Path(uri);
        FSDataInputStream in = fs.open(path);
        BufferedReader bf=new BufferedReader(new InputStreamReader(in));
        String line = null;
        while ((line = bf.readLine()) != null) {
            String[] s=line.split("\t");
            String[] tesFea=s[1].split(",");
            double sum=0.0;
            //计算2张图片的距离
            for(int j=0;j<tesFea.length;j++){
                sum+= (feature[j]-Double.parseDouble(tesFea[j]))*(feature[j]-Double.parseDouble(tesFea[j]));
            }
            if(sum<min){
                min=sum;
                minIndex=s[0];
            }
        }
        fs.close();

        int ind1=name.indexOf("person");
        contex.write(new Text(name.substring(ind1,ind1+11)),new Text(minIndex.substring(ind1,ind1+11)+":"+min));

    }

}
