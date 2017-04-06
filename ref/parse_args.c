#define _GNU_SOURCE
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <unistd.h>

#include "parse_args.h"
#include "params.h"

const static size_t DEFAULT_NUM_POINTS      = 100000;
const static size_t DEFAULT_NCRIT           = 200;
const static int DEFAULT_NUM_TERMS          = 10;
const static TYPE DEFAULT_THETA           = 0.5;
const static size_t DEFAULT_NUM_SAMPLES     = 100;

void init_defaults(t_fmm_options* options)
{
    options->num_points     = DEFAULT_NUM_POINTS;
    options->ncrit          = DEFAULT_NCRIT;
    options->num_terms      = DEFAULT_NUM_TERMS;
    options->theta          = DEFAULT_THETA;
    options->num_samples    = DEFAULT_NUM_SAMPLES;
}

void parse_variables(char c, char* optarg, t_fmm_options* options)
{
    switch(c)
    {
        case 'n':
            options->num_points = atoi(optarg);
            break;
        case 'c':
            options->ncrit = atoi(optarg);
            break;
        case 't':
            options->num_terms = atoi(optarg);
            break;
        case 'e':
            options->theta = atof(optarg);
            break;
        case 'm':
            options->num_samples = atoi(optarg);
            break;
        case '?':
            fprintf(stderr, "error - %c not recognised or missing value\n", optopt);
            break;
    }
}

// this function won't work with arugments that have no value (i.e. flags)
void parse_input_file(const char* fn, t_fmm_options* options)
{
    FILE* f = fopen(fn, "r");
    if (f == NULL)
    {
        fprintf(stderr, "could not open input file\n");
        abort();
    }
    char* buf = NULL;
    size_t n = 0;
    ssize_t read = 0;

    char arg[128];
    while((read = getline(&buf, &n, f)) != -1)
    {
        for (ssize_t i = 0; i < read; ++i)
        {
            if (buf[i] == '-' || buf[i] == ' ') continue;
            else if (isalpha(buf[i]))
            {
                char arg_iden = buf[i];
                i++;
                while (buf[i] == ' ') ++i;
                if (!isdigit(buf[i])) 
                {
                    fprintf(stderr, "arg %c has no value, buf = %c\n", arg_iden, buf[i]);
                    --i;
                    continue;
                }
                sscanf(&buf[i], "%s", arg);
                parse_variables(arg_iden, arg, options);
            }
        }
    }

    free(buf);
    fclose(f);
}

void print_help() { printf("printing help-----\n");}

void parse_fmm_args(int argc, char** argv, t_fmm_options* options)
{
    static struct option long_options[] = 
    {
        {"help",        no_argument,        0,  'h'},
        {"particles",   required_argument,  0,  'n'},
        {"crit",        required_argument,  0,  'c'},
        {"terms",       required_argument,  0,  't'},
        {"theta",       required_argument,  0,  'e'},
        {"samples",     required_argument,  0,  'm'},
        {"input",       required_argument,  0,  'i'},
    };

    int c;
    opterr = 0;

    char* input_filename;
    int need_to_parse_file = 0;

    // temporaries used to store command line args
    t_fmm_options tmp_options;
    init_defaults(&tmp_options);
    
    // initialise variables with defaults
    init_defaults(options);

    while ((c = getopt_long(argc, argv, "hn:c:t:e:m:d:i:", long_options, NULL)) != -1)
    {
        switch(c)
        {
            case 'h':
                print_help();
                exit(0);
            case 'i':
                input_filename = optarg;
                need_to_parse_file = 1;
                break;
            default:
                parse_variables(c, optarg, &tmp_options);
                break;
        }
    }

    // if there's an input file parse it, override defaults
    if (need_to_parse_file) parse_input_file(input_filename, options);

    // override defaults and input file if variable read from command line
    options->num_points         = (tmp_options.num_points    == DEFAULT_NUM_POINTS      ) ? options->num_points     : tmp_options.num_points;
    options->ncrit              = (tmp_options.ncrit         == DEFAULT_NCRIT           ) ? options->ncrit          : tmp_options.ncrit;
    options->num_terms          = (tmp_options.num_terms     == DEFAULT_NUM_TERMS       ) ? options->num_terms      : tmp_options.num_terms;
    options->theta              = (tmp_options.theta         == DEFAULT_THETA           ) ? options->theta          : tmp_options.theta;
    options->num_samples        = (tmp_options.num_samples   == DEFAULT_NUM_SAMPLES     ) ? options->num_samples    : tmp_options.num_samples;
}

void check_args(t_fmm_options* options)
{
    int err = 0;
    if (options->ncrit > options->num_points) {err = 1; fprintf(stderr, "error - bin size greater than no. of particles\n"); }
    if (options->num_samples > options->num_points) {err = 1; fprintf(stderr, "error - no. of samples greater than no. particles"); }
    if (options->theta < 0.0) {err = 1; fprintf(stderr, "error - negative theta value\n"); }
    if (err) exit(1);
}

void print_args(t_fmm_options* options)
{
    printf("-------------- FMM ARGS --------------\n");
    printf("%-20s"   "%zu (%d^%.0f)\n"  , "no. particles",    options->num_points, 10, log10((double)options->num_points));
    printf("%-20s"   "%zu\n"            , "bin size",         options->ncrit);
    printf("%-20s"   "%d\n"             , "no. terms",        options->num_terms);
    printf("%-20s"   "%.2f\n"           , "theta",            options->theta);
    printf("%-20s"   "%zu\n"            , "no. samples",      options->num_samples);
    printf("--------------------------------------\n");
}
