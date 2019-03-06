import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.io.BytesWritable;
import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;

public class ImageToSequenceFileMapper extends Mapper<Object, Text, Text, BytesWritable> {
    public void map(Object key, Text value, Context context)throws IOException, InterruptedException {
       //文件路径以换行符分割
        String lines[] = value.toString().split("\n");
        int numLines = lines.length;
        Configuration conf = new Configuration();
        for (int i = 0; i < numLines; i++) {
            //获取文件路径
            String uri = lines[i].toString();
            FileSystem fs = FileSystem.get(conf);
            FSDataInputStream in = null;
            try {
                //读取图像内容
                in = fs.open(new Path(uri));
                in.seek(0);
                BytesWritable v = new BytesWritable(org.apache.commons.io.IOUtils.toByteArray(in));
                String s = lines[i].toString();
                value.set(s);
                context.write(value, v);
            } finally {
                org.apache.hadoop.io.IOUtils.closeStream(in);
            }

        }

    }

}
