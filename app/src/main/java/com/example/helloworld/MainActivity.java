package com.example.helloworld;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.helloworld.ml.ModelDefect;

import org.tensorflow.lite.DataType;

import org.tensorflow.lite.TensorFlowLite;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;


import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView result;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        imageView = findViewById(androidx.appcompat.R.id.image);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }


    public void ClassifyImage(Bitmap image){
        try {

            ModelDefect model = ModelDefect.newInstance(getApplicationContext());
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 200, 200, 1}, DataType.FLOAT32);


            ImageProcessor imageProcessor = new ImageProcessor.Builder().add(new ResizeOp(200, 200, ResizeOp.ResizeMethod.BILINEAR)).add(new TransformToGrayscaleOp()).build();
            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(image);
            tensorImage = imageProcessor.process(tensorImage);

            ByteBuffer byteBuffer = tensorImage.getBuffer();

            // Creates inputs for reference.
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelDefect.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();


            String[] classes = {"OK", "NOK"};

            int[] resultArr = outputFeature0.getIntArray();

            result.setText(classes[resultArr[0]]);



            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    //RECEIVING IMAGE FROM GALLERY OR CAMERA
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode == RESULT_OK){
            if(requestCode == 3){    //we get picture from camera
                
                Bitmap image = (Bitmap) data.getExtras().get("data");
                //int dimension = Math.min(image.getWidth(), image.getHeight());
                //image = ThumbnailUtils.extractThumbnail(image, dimension,dimension);
                //imageView.setImageBitmap(image);
                //image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);

                //convert to grayscale - then we classify

                ClassifyImage(image);

            }
            else{//get picture from gallery
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                }
                catch (IOException e){
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                //image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }


    //PASSING IMAGE TO CAMERA


}