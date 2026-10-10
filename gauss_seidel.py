"""
Kasus: Analisis mesh rangkaian resistor 3 loop (arus loop I1, I2, I3)

Resistor (ohm): R_a=5 (loop1), R_b=8 (loop2), R_c=10 (loop3)
Resistor bersama: R12=6 (loop1-2), R23=7 (loop2-3), R13=3 (loop1-3)
Sumber tegangan (V): V1=12, V2=5, V3=9

Persamaan KVL:
  14*I1 - 6*I2 - 3*I3 = 12
  -6*I1 + 21*I2 - 7*I3 = 5
  -3*I1 - 7*I2 + 20*I3 = 9
"""
import numpy as np

A = np.array([[14.0, -6.0, -3.0],
              [-6.0, 21.0, -7.0],
              [-3.0, -7.0, 20.0]])
b = np.array([12.0, 5.0, 9.0])

x0 = np.zeros(3)   # tebakan awal
tol = 1e-6         # toleransi (norma-inf selisih antar iterasi)
max_iter = 100     # iterasi maksimum


def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(b)
    x = x0.copy()
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)  # x[j<i] sudah baru
            x[i] = (b[i] - s) / A[i, i]
        err = np.max(np.abs(x - x_old))
        print(f"{k:3d} | " + " | ".join(f"{v:10.6f}" for v in x) + f" | {err:.3e}")
        if err < tol:
            return x, k
    return x, max_iter


if __name__ == "__main__":
    dominan = all(abs(A[i, i]) > np.sum(np.abs(A[i])) - abs(A[i, i]) for i in range(3))
    print("METODE GAUSS-SEIDEL")
    print(f"Tebakan awal = {x0}, toleransi = {tol}, maks iterasi = {max_iter}")
    print(f"Dominan diagonal: {dominan}\n")
    print(" k |         I1 |         I2 |         I3 | error")
    x, k = gauss_seidel(A, b, x0, tol, max_iter)
    print(f"\nKonvergen pada iterasi ke-{k}")
    print(f"Solusi Gauss-Seidel : I1={x[0]:.6f} A, I2={x[1]:.6f} A, I3={x[2]:.6f} A")
    print(f"Solusi eksak        : {np.linalg.solve(A, b)}")
