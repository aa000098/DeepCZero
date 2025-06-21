#include <curl/curl.h>
#include <zlib.h>
#include <fstream>
#include <filesystem>


std::string get_file(const std::string& url) {
    const char* home = std::getenv("HOME");
    if (!home) throw std::runtime_error("HOME environment variable not set");

    std::filesystem::path dir_path = std::string(home) + "/.deepczero/datasets";
    std::filesystem::create_directories(dir_path);

    std::string filename = url.substr(url.find_last_of("/") + 1);
    std::filesystem::path file_path = dir_path / filename;

    if (std::filesystem::exists(file_path)) return file_path.string();

    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("Failed to init curl");

    FILE* fp = fopen(file_path.c_str(), "wb");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    CURLcode res = curl_easy_perform(curl);
    fclose(fp);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK)
        throw std::runtime_error("Download failed: " + url);

    return file_path.string();
}

std::string gunzip_file(const std::string& gz_path) {
    std::string out_path = gz_path.substr(0, gz_path.find_last_of('.'));  // remove .gz

    if (std::filesystem::exists(out_path))
        return out_path;

    gzFile infile = gzopen(gz_path.c_str(), "rb");
    if (!infile) throw std::runtime_error("Cannot open gzip file: " + gz_path);

    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot create file: " + out_path);

    char buffer[8192];
    int num_read;
    while ((num_read = gzread(infile, buffer, sizeof(buffer))) > 0)
        out.write(buffer, num_read);

    gzclose(infile);
    return out_path;
}
