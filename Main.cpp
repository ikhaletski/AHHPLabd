#include "MatrixService.h"
#include <Windows.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>


using namespace std;

ULONGLONG SSETime;
ULONGLONG vectTime;
ULONGLONG noVectTime;
ULONGLONG endTime;
ULONGLONG startTime;

double**** matrixA;
double**** matrixB;
double**** matrixC;
double**** matrixD;

double**** getMatrix(int, int, int, int);
void matrixFilling(double****, int, int, int, int);
double**** mul(double****, double****, int, int, int, int);
double**** mulAutoVect(double****, double****, int, int, int, int);
double**** mulSSE(double****, double****, int, int, int, int);
bool equals(double****, double****, int, int, int, int);

int main() {
	MatrixService matrixService;

	matrixA = getMatrix(8, 8, 500, 500);
	matrixB = getMatrix(8, 8, 500, 500);

	matrixFilling(matrixA, 8, 8, 500, 500);
	matrixFilling(matrixB, 8, 8, 500, 500);

	startTime = GetTickCount64();

	matrixC = mulAutoVect(matrixA, matrixB, 8, 8, 150, 150);
	endTime = GetTickCount64();
	vectTime = endTime - startTime;

	startTime = GetTickCount64();
	matrixD = mulSSE(matrixA, matrixB, 8, 8, 150, 150);
	endTime = GetTickCount64();
	SSETime = endTime - startTime;

	startTime = GetTickCount64();
	mul(matrixA, matrixB, 8, 8, 150, 150);
	endTime = GetTickCount64();
	noVectTime = endTime - startTime;

	cout << "Not Victorized Multiple: " << noVectTime << "." << endl;
	cout << "Victorization Multiple: " << vectTime << "." << endl;
	cout << "SSE Multiple: " << SSETime << ".\n" << endl;
	cout << "Not Vectorized/Vectorized: " << (double)noVectTime / (double)vectTime << "." << endl;
	cout << "Not Vectorized/SSE: " << (double)noVectTime / (double)SSETime << ".\n" << endl;

	if (equals(matrixC, matrixD, 8, 8, 150, 150)) cout << "equals" << endl;
	else  cout << "not equals" << endl;

	delete(matrixA);
	delete(matrixB);
	delete(matrixC);
	delete(matrixD);

	system("pause");
	return 0;
}

double**** getMatrix(int matrixHeight, int matrixWidth, int subMatrixHeight, int subMatrixWidth) {
	double**** matrix = nullptr;
	matrix = new double*** [matrixHeight];

	for (int i = 0; i < matrixHeight; i++) {
		matrix[i] = new double** [matrixWidth];

		for (int j = 0; j < matrixWidth; j++) {
			matrix[i][j] = new double* [subMatrixHeight];

			for (int k = 0; k < subMatrixHeight; k++) {
				matrix[i][j][k] = new double[subMatrixWidth];

				for (int m = 0; m < subMatrixWidth; m++) {
					matrix[i][j][k][m] = 0;
				}
			}
		}
	}
	return matrix;
}

void matrixFilling(double**** matrix, int matrixHeight, int matrixWidth, int subMatrixHeight, int subMatrixWidth) {
	srand((unsigned)time(NULL));

	for (int i = 0; i < matrixHeight; i++) {
		for (int j = 0; j < matrixWidth; j++) {
			for (int k = 0; k < subMatrixHeight; k++) {
				for (int m = 0; m < subMatrixWidth; m++) {
					matrix[i][j][k][m] = rand() % 100;
				}
			}
		}
	}
}

double**** mulAutoVect(double**** matrixA, double**** matrixB, int matrixHeight, int matrixWidth, int subMatrixHeight, int subMatrixWidth) {
	double**** matrixC = nullptr;

	matrixC = getMatrix(matrixHeight, matrixWidth, subMatrixHeight, subMatrixWidth);

	for (int m = 0; m < matrixHeight; m++) {
		for (int n = 0; n < matrixWidth; n++) {
			for (int i = 0; i < subMatrixHeight; i++) {
				for (int j = 0; j < subMatrixWidth; j++) {
					for (int k = 0; k < subMatrixWidth; k++) {
						matrixC[m][n][i][k] += matrixA[m][n][i][j] * matrixB[m][n][j][k];
					}

				}
			}
		}
	}

	return matrixC;
}

double**** mul(double**** matrixA, double**** matrixB, int matrixHeight, int matrixWidth, int subMatrixHeight, int subMatrixWidth) {
	double**** matrixC = nullptr;

	matrixC = getMatrix(matrixHeight, matrixWidth, subMatrixHeight, subMatrixWidth);

	for (int m = 0; m < matrixHeight; m++) {
		for (int n = 0; n < matrixWidth; n++) {
			for (int i = 0; i < subMatrixHeight; i++) {
				for (int j = 0; j < subMatrixWidth; j++) {
#pragma loop(no_vector)
					for (int k = 0; k < subMatrixWidth; k++) {
						matrixC[m][n][i][k] += matrixA[m][n][i][j] * matrixB[m][n][j][k];
					}

				}
			}
		}
	}

	return matrixC;
}

double**** mulSSE(double**** matrixA, double**** matrixB, int matrixHeight, int matrixWidth, int subMatrixHeight, int subMatrixWidth) {
	double**** matrixC = nullptr;
	double tempA = 0;
	double* tempB = nullptr;
	double* tempC = nullptr;

	__m128d reg0;
	__m128d reg1;
	__m128d reg2;

	matrixC = getMatrix(matrixHeight, matrixWidth, subMatrixHeight, subMatrixWidth);

	for (int m = 0; m < matrixHeight; m++) {
		for (int n = 0; n < matrixWidth; n++) {
			for (int i = 0; i < subMatrixHeight; i++)
			{
				tempC = matrixC[m][n][i];

				for (int j = 0; j < subMatrixWidth; j++)
				{
					tempA = matrixA[m][n][i][j];
					tempB = matrixB[m][n][j];
					reg1 = _mm_set1_pd(tempA);

					for (int k = 0; k < subMatrixWidth; k += 2)
					{
						reg0 = _mm_load_pd(tempC + k);
						reg2 = _mm_load_pd(tempB + k);
						reg0 = _mm_add_pd(reg0, _mm_mul_pd(reg1, reg2));
						_mm_store_pd(tempC + k, reg0);
					}
				}
			}
		}
	}

	return matrixC;
}

bool equals(double**** matrixA, double**** matrixB, int matrixHeight, int matrixWidth, int subMatrixHeight, int subMatrixWidth) {
	for (int i = 0; i < matrixHeight; i++) {
		for (int j = 0; j < matrixWidth; j++) {
			for (int k = 0; k < subMatrixHeight; k++) {
				for (int l = 0; l < subMatrixWidth; l++) {
					if (matrixA[i][j][k][l] != matrixB[i][j][k][l]) {
						return false;
					}
				}
			}
		}
	}

	return true;
}