#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>

#include "parse_args.h"
#include "params.h"

const static size_t DEFAULT_NUM_POINTS      = 100000;
const static size_t DEFAULT_NCRIT           = 200;
const static int DEFAULT_NUM_TERMS          = 10;
const static TYPE DEFAULT_THETA             = 0.5;
const static size_t DEFAULT_NUM_SAMPLES     = 100;

void init_defaults(t_fmm_params* params)
{
    params->num_points     = DEFAULT_NUM_POINTS;
    params->ncrit          = DEFAULT_NCRIT;
    params->num_terms      = DEFAULT_NUM_TERMS;
    params->theta          = DEFAULT_THETA;
    params->num_samples    = DEFAULT_NUM_SAMPLES;
}

void parse_variables(char c, char* optarg, t_fmm_params* params)
{
    switch(c)
    {
        case 'n':
            params->num_points = atoi(optarg);
            break;
        case 'c':
            params->ncrit = atoi(optarg);
            break;
        case 't':
            params->num_terms = atoi(optarg);
            break;
        case 'e':
            params->theta = atof(optarg);
            break;
        case 'm':
            params->num_samples = atoi(optarg);
            break;
        case '?':
            fprintf(stderr, "error - %c not recognised or missing value\n", optopt);
            break;
    }
}

int get_line(char **lineptr, size_t *n, FILE *stream)
{
    static char line[256];
    char *ptr;
    unsigned int len;

    if (lineptr == NULL || n == NULL || ferror(stream) || feof(stream))
        return -1;

    char* ret = fgets(line,256,stream);
    if (ret == NULL)
        return -1;

    ptr = strchr(line,'\n');   
    if (ptr) *ptr = '\0';
    len = strlen(line);

    if ((len+1) < 256)
    {
        ptr = (char*)realloc(*lineptr, 256);
        if (ptr == NULL) return(-1);
        *lineptr = ptr;
        *n = 256;
    }

    strcpy(*lineptr,line); 
    return(len);
}

// this function won't work with arugments that have no value (i.e. flags)
void parse_input_file(const char* fn, t_fmm_params* params)
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
    while((read = get_line(&buf, &n, f)) != -1)
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
                parse_variables(arg_iden, arg, params);
            }
        }
    }

    free(buf);
    fclose(f);
}

void print_help() 
{
    printf("\nMiniFMM\n");
    printf("Usage: ./fmm [OPTIONS]\n");
    printf("Options: (symbol, name, default value, description\n");
    printf("    -h  --help                  Print this message\n");
    printf("    -n  --particles     10^5    Number of particles in problem\n");
    printf("    -c  --crit          200     Maximum number of particles per leaf node\n");
    printf("    -t  --terms         10      Number of multipole terms in expansion\n");
    printf("    -e  --theta         0.5     Ratio of node size to distance at which to approximate\n");
    printf("    -m  --samples       100     Number of samples to test solution against\n");
    printf("    -i  --input                 Input file name\n");
    printf("Notes:\n");
    printf("    - Command line arguments override parameters in input file\n");
    printf("\n");
}

void parse_fmm_args(int argc, char** argv, t_fmm_params* params)
{
    static struct option long_params[] = 
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
    t_fmm_params tmp_params;
    init_defaults(&tmp_params);
    
    // initialise variables with defaults
    init_defaults(params);

    while ((c = getopt_long(argc, argv, "hn:c:t:e:m:d:i:", long_params, NULL)) != -1)
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
                parse_variables(c, optarg, &tmp_params);
                break;
        }
    }

    // if there's an input file parse it, override defaults
    if (need_to_parse_file) parse_input_file(input_filename, params);

    // override defaults and input file if variable read from command line
    params->num_points         = (tmp_params.num_points    == DEFAULT_NUM_POINTS      ) ? params->num_points     : tmp_params.num_points;
    params->ncrit              = (tmp_params.ncrit         == DEFAULT_NCRIT           ) ? params->ncrit          : tmp_params.ncrit;
    params->num_terms          = (tmp_params.num_terms     == DEFAULT_NUM_TERMS       ) ? params->num_terms      : tmp_params.num_terms;
    params->theta              = (tmp_params.theta         == DEFAULT_THETA           ) ? params->theta          : tmp_params.theta;
    params->num_samples        = (tmp_params.num_samples   == DEFAULT_NUM_SAMPLES     ) ? params->num_samples    : tmp_params.num_samples;

    // if we have more samples than actual points, set no. samples equal to no. points
    if (params->num_samples > params->num_points) params->num_samples = params->num_points;
}

void check_args(t_fmm_params* params)
{
    int err = 0;
    if (params->ncrit > params->num_points) {err = 1; fprintf(stderr, "error - bin size greater than no. of particles\n"); }
    if (params->theta < 0.0) {err = 1; fprintf(stderr, "error - negative theta value\n"); }
    if (err) exit(1);
}

void print_args(t_fmm_params* params)
{
    printf("-------------- FMM ARGS --------------\n");
    printf("%-20s"   "%zu (%d^%.0f)\n"  , "no. particles",    params->num_points, 10, log10((double)params->num_points));
    printf("%-20s"   "%zu\n"            , "bin size",         params->ncrit);
    printf("%-20s"   "%d\n"             , "no. terms",        params->num_terms);
    printf("%-20s"   "%.2f\n"           , "theta",            params->theta);
    printf("%-20s"   "%zu\n"            , "no. samples",      params->num_samples);
    printf("--------------------------------------\n");
}

