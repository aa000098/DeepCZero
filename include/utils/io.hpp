#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <iostream>

#include <curl/curl.h>

static std::string _expand_user(const std::string& p) {
    if (!p.empty() && p[0] == '~') {
#ifdef _WIN32
        const char* home = std::getenv("USERPROFILE");
#else
        const char* home = std::getenv("HOME");
#endif
        if (!home) throw std::runtime_error("HOME/USERPROFILE not set");
        if (p.size() == 1) return std::string(home);
        if (p[1] == '/' || p[1] == '\\')
            return (std::filesystem::path(home) / p.substr(2)).string();
    }
    return p;
}

static std::string _basename_from_url(const std::string& url) {
    // strip query/fragment
    size_t end = url.find_first_of("?#");
    std::string path = (end == std::string::npos) ? url : url.substr(0, end);
    // last segment after '/'
    size_t pos = path.find_last_of('/');
    std::string name = (pos == std::string::npos) ? path : path.substr(pos + 1);
    if (name.empty()) name = "download.bin";
    return name;
}

static size_t _write_file(void* ptr, size_t size, size_t nmemb, void* stream) {
    return std::fwrite(ptr, size, nmemb, static_cast<FILE*>(stream));
}

static int _progress_cb(void* /*clientp*/,
                        curl_off_t dltotal, curl_off_t dlnow,
                        curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    if (dltotal > 0) {
        double pct = (double)dlnow * 100.0 / (double)dltotal;
        static int last = -1;
        int ipct = (int)pct;
        if (ipct != last) {
            last = ipct;
            std::cerr << "\r  " << ipct << "%";
            std::cerr.flush();
        }
    }
    return 0; // 0 = continue
}

// 환경변수 DEEPCZERO_CACHE 가 있으면 그 경로 사용, 없으면 "~/.deepczero"
static std::string _default_cache_dir() {
    const char* env = std::getenv("DEEPCZERO_CACHE");
    if (env && *env) return env;
#ifdef _WIN32
/*
    const char* appdata = std::getenv("APPDATA");
    if (appdata && *appdata) {
        return (std::filesystem::path(appdata) / "DeepCZero" / "cache").string();
    }
*/
    const char* userp = std::getenv("USERPROFILE");
    if (userp && *userp) {
        return (std::filesystem::path(userp) / ".deepczero").string();
    }
    return ".\\.deepczero";
#else
    return "~/.deepczero";
#endif
}

/**
 * URL에서 파일을 캐시에 내려받고, 절대 경로를 반환한다.
 * 이미 존재하면 다운로드 생략.
 *
 * @param url       다운로드 URL
 * @param file_name 빈 문자열이면 URL의 마지막 세그먼트를 파일명으로 사용
 */
static std::string get_file(std::string url, std::string file_name = "") {
    // 캐시 루트
    std::string cache = _default_cache_dir();
    cache = _expand_user(cache);

    std::filesystem::path cache_dir(cache);
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec); // 이미 있으면 OK
    if (ec) throw std::runtime_error("Failed to create cache dir: " + cache_dir.string());

    // 파일명 결정
    if (file_name.empty()) file_name = _basename_from_url(url);
    std::filesystem::path out_path = cache_dir / file_name;

    // 이미 있으면 그대로 반환
    if (std::filesystem::exists(out_path)) {
        return std::filesystem::absolute(out_path).string();
    }

    // 임시 파일 경로
    std::filesystem::path tmp_path = out_path;
    tmp_path += ".tmp";

    std::cerr << "Downloading: " << file_name << "\n";

    // libcurl 세팅
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init() failed");

    FILE* fp = std::fopen(tmp_path.string().c_str(), "wb");
    if (!fp) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("Failed to open temp file: " + tmp_path.string());
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L); // HTTP 4xx/5xx -> error
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _write_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
#if LIBCURL_VERSION_NUM >= 0x072000 // 7.32.0
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, _progress_cb);
#else
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, _progress_cb);
#endif
    // 타임아웃 (원하면 조절)
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L); // 무제한

    CURLcode res = curl_easy_perform(curl);

    std::fclose(fp);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::filesystem::remove(tmp_path, ec);
        throw std::runtime_error(std::string("Download failed: ") + curl_easy_strerror(res));
    }

    // 원자적 교체 (같은 파일시스템 내)
    std::filesystem::rename(tmp_path, out_path, ec);
    if (ec) {
        // rename 실패 시 temp 삭제 시도
        std::filesystem::remove(tmp_path, ec);
        throw std::runtime_error("Failed to move temp to target: " + out_path.string());
    }

    std::cerr << " Done\n";
    return std::filesystem::absolute(out_path).string();
}


