import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter FastAPI Example',
      home: HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  Future<Map<String, dynamic>> fetchData() async {
    final response = await http.get(Uri.parse('http://20.10.6.13:8000 /'));

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to load data');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Flutter FastAPI Example'),
      ),
      body: FutureBuilder(
        future: fetchData(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else {
            // Use fetched data from FastAPI backend here
            return Center(child: Text('Data from FastAPI: ${snapshot.data}'));
          }
        },
      ),
    );
  }
}