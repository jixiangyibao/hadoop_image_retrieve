import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;

public class imageRetrieval {
    public static void main(String[] args)throws Exception {
        //job1
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        Job job1 = new Job(conf, "job1");
        job1.setJarByClass(imageRetrieval.class);
        job1.setMapperClass(ImageToSequenceFileMapper.class);
        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(BytesWritable.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(BytesWritable.class);
        //job1.setNumReduceTasks(10);
       // LazyOutputFormat.setOutputFormatClass(job1, SequenceFileOutputFormat.class);
        //MultipleOutputs.addNamedOutput(job1, "n", SequenceFileOutputFormat.class, Text.class, BytesWritable.class);
        FileInputFormat.setMinInputSplitSize(job1, 1);// 设定文件分片，这样才能让多个mapper和reducer实际用起来
        FileInputFormat.setMaxInputSplitSize(job1, 10485760);
        //设置输入输出路径
        FileInputFormat.addInputPath(job1, new Path(otherArgs[0]));
        //FileOutputFormat.setOutputPath(job1, new Path(otherArgs[1]));
        job1.setOutputFormatClass(SequenceFileOutputFormat.class);
        SequenceFileOutputFormat.setOutputPath(job1, new Path(otherArgs[1]));
        job1.waitForCompletion(true);

        //job2
        Configuration conf2 = new Configuration();
        Job job2 = new Job(conf2, "job2");
        job2.setJarByClass(imageRetrieval.class);
        job2.setMapperClass(FeatExtrLOMO.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        FileInputFormat.setMinInputSplitSize(job2, 1);// 设定文件分片，这样才能让多个mapper和reducer实际用起来
        FileInputFormat.setMaxInputSplitSize(job2, 10485760);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        job2.setInputFormatClass(SequenceFileInputFormat.class);
        FileInputFormat.setInputPaths(job2, new Path(otherArgs[1]));
        FileOutputFormat.setOutputPath(job2, new Path(otherArgs[2]));
        job2.waitForCompletion(true);

        Configuration conf3 = new Configuration();
        Job job3 = new Job(conf3, "job3");
        job3.setJarByClass(imageRetrieval.class);
        job3.setMapperClass(knnMapper.class);
        job3.setReducerClass(knnReducer.class);
        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(Text.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);
        FileInputFormat.setInputPaths(job3, new Path(otherArgs[2]));
        FileOutputFormat.setOutputPath(job3, new Path(otherArgs[3]));


        System.exit(job3.waitForCompletion(true) ? 0 : 1);
    }
}
