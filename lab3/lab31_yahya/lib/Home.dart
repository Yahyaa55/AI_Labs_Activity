import 'package:flutter/material.dart';   //material : ANDROID , cupertino : IPHONE
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';

class Home extends StatefulWidget {
  const Home({Key? key}) : super(key: key);

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  File? _image;
  List? _output;

  Future<void> _takePhoto() async {
    //Pick an image from camera or gallery
    final XFile? image =
    await ImagePicker().pickImage(source: ImageSource.camera);
    if (image == null) {
      return null;
    } else {
      setState(() {
        _image = File(image.path);
        detectimage(_image!);
      });
    }
    ;
    //detectimage(_image!);
  }
  loadmodel() async {
    await Tflite.loadModel(
        model: 'assets/model.tflite', labels: 'assets/labels.txt');
  }
  @override
  void initState() {
    super.initState();
    loadmodel().then((value) {
      setState(() {});
    });
  }
  detectimage(File image) async {
    print("okkkkkkkkkkkkkk" + image.path);
    var prediction = await Tflite.runModelOnImage(
        path: image.path,
        numResults: 2,
        threshold: 0.6,
        imageMean: 127.5,
        imageStd: 127.5);
    setState(() {
      _output = prediction!;
      print(_output);
    });
  }
  @override
  void dispose() {
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text("Face mask detection"),
        ),
        body: Center(
            child: Column(
              children: [
                _image == null
                    ? Text('No image selected')
                    : Image.file(
                  _image!,
                  width: 300,
                  height: 300,
                ),
                _output != null
                    ? Text((_output![0]['label']).toString().substring(2),
                    style: TextStyle(fontSize: 28))
                    : Text(''),
                _output != null
                    ? Text('Confidence: ' + (_output![0]['confidence']).toString(),
                    style: TextStyle(fontSize: 28))
                    : Text('')
              ],
            )),
        floatingActionButton: FloatingActionButton(
            onPressed: () {
              _takePhoto();
              print('okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk');
            },
            child: Icon(Icons.camera_alt),
            ),
        );
    }
}
