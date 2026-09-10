//
// Created by lfc on 2021/3/1.
//

#ifndef SRC_GAZEBO_CSV_READER_HPP
#define SRC_GAZEBO_CSV_READER_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class CsvReader {
 public:
    static bool ReadCsvFile(std::string file_name, std::vector<std::vector<double>>& datas) {
        std::fstream file_stream;
        file_stream.open(file_name, std::ios::in);
        if (file_stream.is_open()) {
            std::string header;
            std::getline(file_stream, header, '\n');
            std::string line_str;
            while (std::getline(file_stream, line_str, '\n')) {
                // Skip blank / trailing lines; empty stod() used to spam logs
                // and left the plugin in a fragile state after huge CSVs.
                if (line_str.empty() ||
                    line_str.find_first_not_of(" \t\r") == std::string::npos) {
                    continue;
                }
                std::stringstream line_stream(line_str);
                std::vector<double> data;
                try {
                    std::string value;
                    while (std::getline(line_stream, value, ',')) {
                        if (value.empty()) {
                            continue;
                        }
                        data.push_back(std::stod(value));
                    }
                } catch (...) {
                    std::cerr << "cannot convert str:" << line_str << "\n";
                    continue;
                }
                if (!data.empty()) {
                    datas.push_back(data);
                }
            }
            std::cerr << "data size:" << datas.size() << "\n";
            return true;
        } else {
            std::cerr << "cannot read csv file!" << file_name << "\n";
        }
        return false;
    }
};

#endif  // SRC_GAZEBO_CSV_READER_HPP
