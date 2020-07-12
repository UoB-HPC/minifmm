#pragma once

#include <getopt.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include <fmm.hh>

void print_help() { printf("help\n"); }

template <class T>
void parse_var(const char c, const char* optarg, FMM<T>* fmm)
{
  switch (c) {
    case 'n':
      fmm->num_points = std::atoi(optarg);
      break;
    case 'c':
      fmm->ncrit = std::atoi(optarg);
      break;
    case 't':
      fmm->num_terms = std::atoi(optarg);
      break;
    case 'e':
      fmm->theta = std::atof(optarg);
      break;
    case 'm':
      fmm->num_samples = std::atoi(optarg);
      break;
    case 'p':
      fmm->dist = FMM<T>::Dist::Plummer;
      break;
    case 'u':
      fmm->dist = FMM<T>::Dist::Uniform;
      break;
    case '?':
      fprintf(stderr, "error - %c not recognised or missing value\n", optopt);
      break;
  }
}

template <class T>
void parse_input_file(const char* ifile, FMM<T>* fmm)
{
  std::ifstream ifs(ifile);
  if (!ifs.is_open()) {
    std::cerr << "error: could not open input file - " << ifile << std::endl;
    std::exit(1);
  }

  std::vector<std::string> lines;

  std::string line;
  while (std::getline(ifs, line)) {
    std::stringstream ss(line);
    while (std::getline(ss, line, ' ')) {
      lines.push_back(line);
    }
  }

  int fargc = lines.size() + 1;
  char** fargv = (char**)malloc(sizeof(char*) * fargc);
  for (size_t i = 0; i < lines.size(); ++i) {
    fargv[i + 1] = (char*)malloc(sizeof(char) * lines[i].size());
    strcpy(fargv[i + 1], lines[i].c_str());
  }
  parse_args(fargc, fargv, fmm, true);
}

template <class T>
void parse_args(int argc, char** argv, FMM<T>* fmm, bool nested)
{
  const static struct option long_params[] = {
      {"help", no_argument, NULL, 'h'},
      {"npart", required_argument, NULL, 'n'},
      {"ncrit", required_argument, NULL, 'c'},
      {"nterms", required_argument, NULL, 't'},
      {"theta", required_argument, NULL, 'e'},
      {"nsamp", required_argument, NULL, 'm'},
      {"ifile", required_argument, NULL, 'i'},
      {"plummer", no_argument, NULL, 'p'},
      {"uniform", no_argument, NULL, 'u'},
  };

  int c;
  optind = 1;
  opterr = 0;
  while ((c = getopt_long(argc, argv, "hpun:c:t:e:m:d:i:", long_params, NULL)) !=
         -1) {
    switch (c) {
      case 'h':
        print_help();
        // TODO check this leads to correct destruction.
        std::exit(0);
      case 'i':
        // stop input file arg being used inside an input file
        if (!nested) {
          const char* ifile = optarg;
          parse_input_file(ifile, fmm);
          // TODO fix this
          return;
        }
        break;
      default:
        parse_var(c, optarg, fmm);
        break;
    }
  }
}

template <class T>
void read_input(int argc, char** argv, FMM<T>* fmm)
{
  const char* dist_strings[FMM<T>::Dist::NumDist] = {"Uniform", "Plummer"};
  fmm->num_points = 1000;
  fmm->ncrit = 20;
  fmm->num_terms = 4;
  fmm->theta = 0.5;
  fmm->num_samples = 1000;
  fmm->dist = FMM<T>::Dist::Plummer;

  parse_args(argc, argv, fmm, false);

  fmm->num_samples = std::min(fmm->num_samples, fmm->num_points);

  std::cout << "FMM args\n"
            << "Num Points   = " << fmm->num_points << '\n'
            << "NCrit        = " << fmm->ncrit << '\n'
            << "Num Terms    = " << fmm->num_terms << '\n'
            << "Theta        = " << fmm->theta << '\n'
            << "Num Samples  = " << fmm->num_samples << "\n"
            << "Distribution = " << dist_strings[fmm->dist] << "\n\n";
}
