#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <cuda_helper.h>
#include <miner.h>


#define  F(x, y, z) (((x) ^ (y) ^ (z)))
#define FF(x, y, z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define GG(x, y, z) ((z)  ^ ((x) & ((y) ^ (z))))

#define P0(x) x ^ ROTL32(x,  9) ^ ROTL32(x, 17)
#define P1(x) x ^ ROTL32(x, 15) ^ ROTL32(x, 23)

__device__
void sm3_compress2(uint32_t digest[8], unsigned char block[64]){
	uint32_t tt1, tt2, i, t, ss1, ss2, x, y;
	uint32_t w[68];
	uint32_t a = digest[0];
	uint32_t b = digest[1];
	uint32_t c = digest[2];
	uint32_t d = digest[3];
	uint32_t e = digest[4];
	uint32_t f = digest[5];
	uint32_t g = digest[6];
	uint32_t h = digest[7];


	const uint32_t *pblock = (const uint32_t *)block;

	for (i = 0; i<16; i++) {
		w[i] = cuda_swab32(pblock[i]);
	}

	for (i = 16; i<68; i++) {
		x = ROTL32(w[i - 3], 15);
		y = ROTL32(w[i - 13], 7);

		x ^= w[i - 16];
		x ^= w[i - 9];
		y ^= w[i - 6];

		w[i] = P1(x) ^ y;
	}

	for (i = 0; i<64; i++) {

		t = (i < 16) ? 0x79cc4519 : 0x7a879d8a;

		ss2 = ROTL32(a, 12);
		ss1 = ROTL32(ss2 + e + ROTL32(t, i), 7);
		ss2 ^= ss1;

		tt1 = d + ss2 + (w[i] ^ w[i + 4]);
		tt2 = h + ss1 + w[i];

		if (i < 16) {
			tt1 += F(a, b, c);
			tt2 += F(e, f, g);
		}
		else {
			tt1 += FF(a, b, c);
			tt2 += GG(e, f, g);
		}
		d = c;
		c = ROTL32(b, 9);
		b = a;
		a = tt1;
		h = g;
		g = ROTL32(f, 19);
		f = e;
		e = P0(tt2);
	}


	digest[0] ^= a;
	digest[1] ^= b;
	digest[2] ^= c;
	digest[3] ^= d;
	digest[4] ^= e;
	digest[5] ^= f;
	digest[6] ^= g;
	digest[7] ^= h;

}


/***************************************************/
// GPU Hash Function
__global__ void x14_sm3_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	__syncthreads();

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);
		int hashPosition = nounce - startNounce;

		uint32_t digest[8];
		digest[0] = 0x7380166F;
		digest[1] = 0x4914B2B9;
		digest[2] = 0x172442D7;
		digest[3] = 0xDA8A0600;
		digest[4] = 0xA96F30BC;
		digest[5] = 0x163138AA;
		digest[6] = 0xE38DEE4D;
		digest[7] = 0xB0FB0E4E;

		unsigned char *pHash = (unsigned char *)&g_hash[hashPosition << 3];
		sm3_compress2(digest, pHash);

		unsigned char block[64] = {0};

		block[0] = 0x80;
		uint32_t *count = (uint32_t *)(block + 64 - 8);

		count[0] = cuda_swab32(1 >> 23);
		count[1] = cuda_swab32((1 << 9) + (0 << 3));

		sm3_compress2(digest, block);

		uint32_t *outpHash = (uint32_t*)&g_hash[hashPosition << 3]; // [8 * hashPosition];

		for (int i = 0; i < 8; i++)
			outpHash[i] = cuda_swab32(digest[i]);

		for (int i = 8; i < 16; i++)
			outpHash[i] = 0;
	}
}

__host__ void x14_sm3_cpu_init(int thr_id, uint32_t threads)
{
}

// #include <stdio.h>
__host__ void x14_sm3_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	x14_sm3_gpu_hash_64 << <grid, block, shared_size >> >(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
	MyStreamSynchronize(NULL, order, thr_id);
}