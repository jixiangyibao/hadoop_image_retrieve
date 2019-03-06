
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import java.io.IOException;


public class knnReducer extends Reducer<Text, Text, Text, Text> {

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        double minDis=Double.MAX_VALUE;
        String minPerson="";
        for(Text value:values){
            String v=value.toString();
            int index=v.indexOf(":");
            double dist=Double.parseDouble(v.substring(index+1));
            if(dist<minDis){
                minDis=dist;
                minPerson=v.substring(0,index);
            }

        }
        context.write(key,new Text(minPerson));
    }
}
