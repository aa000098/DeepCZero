#pragma once

#include <string>
#include <curl/curl.h>
#include <zlib.h>
#include <fstream>
#include <filesystem>


std::string get_dataset_file(const std::string& url);

std::string gunzip_file(const std::string& gz_path);

