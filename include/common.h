#ifndef COMMON_H
#define COMMON_H

template<typename T>
auto constexpr log2(T n) {
    return ( (n<2) ? 1 : 1+log2(n/2));
}

template<typename T>
auto constexpr ceil(T a, T b) {
    return (1 + ((a - 1)/b) );
}


template<typename T, typename U>
auto constexpr pow(T base, U exponent) {
    static_assert(std::is_integral<U>(), "exponent must be integral");
    return exponent == 0 ? 1 : base * pow(base, exponent - 1);
}

int mod (int a, int b)
{
   if(b < 0) //you can check for b == 0 separately and do what you want
     return -mod(-a, -b);   
   int ret = a % b;
   if(ret < 0)
     ret+=b;
   return ret;
}

#endif
