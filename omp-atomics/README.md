## Notes

Intel compiler allows for atomics on complex data types. However, it is incredibly slow.

GCC will fail to compile with atomic on complex data type - would need to seperate real and imaginary parts.
Doing this directly through assuming a complex data type is as a struct:

```c
struct
{
    double real, imaginary;
}
```

causes internal compiler errors; GCC itself actually segfaults.
