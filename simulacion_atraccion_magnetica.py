#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulación de la atracción entre imanes con polos iguales mediada por una esfera ferromagnética.
Basado en el artículo: "Attraction between like-poles of two magnets mediated by
a soft ferromagnetic material" - Teles et al. (2024)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors

# Constantes físicas
mu0 = 4 * np.pi * 1e-7  # Permeabilidad del vacío (H/m)

class MagneticForceSimulation:
    def __init__(self, a, rm, dL, dR, B):
        """
        Inicializa la simulación con los parámetros físicos.
        
        Parámetros:
        -----------
        a : float
            Radio de la esfera ferromagnética (m)
        rm : float
            Radio del imán cilíndrico (m)
        dL : float
            Longitud del imán inferior/izquierdo (m)
        dR : float
            Longitud del imán superior/derecho (m)
        B : float
            Campo magnético en la superficie del imán (T)
        """
        self.a = a
        self.rm = rm
        self.dL = dL
        self.dR = dR
        self.B = B
        
        # Calcular cargas magnéticas equivalentes (monopolos)
        self.QL = self.calculate_monopole_charge(dL)
        self.QR = self.calculate_monopole_charge(dR)
        
        print(f"Configuración:\n"
              f"  Radio de esfera: {a*1000:.2f} mm\n"
              f"  Radio de imanes: {rm*1000:.2f} mm\n"
              f"  Longitud imán izquierdo: {dL*1000:.2f} mm\n"
              f"  Longitud imán derecho: {dR*1000:.2f} mm\n"
              f"  Campo B en superficie: {B:.3f} T\n"
              f"  Carga monopolo calculada: {self.QR:.6e} A·m")
    
    def calculate_monopole_charge(self, d):
        """
        Calcula la carga del monopolo equivalente para un imán cilíndrico.
        
        Parámetros:
        -----------
        d : float
            Longitud del imán (m)
            
        Retorna:
        --------
        Q : float
            Carga del monopolo (A·m)
        """
        return (2 * np.pi * self.B * self.rm**2 / mu0) * np.sqrt(1 + (self.rm/d)**2)
    
    def calculate_force(self, zL, zR):
        """
        Calcula la fuerza magnética entre los imanes en presencia de la esfera ferromagnética.
        
        Parámetros:
        -----------
        zL : float
            Distancia desde el centro de la esfera al imán izquierdo (m)
        zR : float
            Distancia desde el centro de la esfera al imán derecho (m)
            
        Retorna:
        --------
        F : float
            Fuerza magnética (N), positiva para repulsión, negativa para atracción
        """
        # Posiciones de las cargas magnéticas (monopolos)
        r1 = np.array([0, 0, self.a + zR])             # Carga QR en imán derecho (cerca de la esfera)
        r2 = np.array([0, 0, self.a + zR + self.dR])   # Carga -QR en imán derecho (lejos de la esfera)
        r3 = np.array([0, 0, -(self.a + zL)])          # Carga QL en imán izquierdo (cerca de la esfera)
        r4 = np.array([0, 0, -(self.a + zL + self.dL)])# Carga -QL en imán izquierdo (lejos de la esfera)
        
        # Magnitudes de las posiciones
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        r3_mag = np.linalg.norm(r3)
        r4_mag = np.linalg.norm(r4)
        
        # Posiciones de las cargas imagen
        r1_im = (self.a**2 / r1_mag**2) * r1
        r2_im = (self.a**2 / r2_mag**2) * r2
        r3_im = (self.a**2 / r3_mag**2) * r3
        r4_im = (self.a**2 / r4_mag**2) * r4
        
        # Magnitudes de las cargas imagen
        q1_im = -self.QR * self.a / r1_mag
        q2_im = self.QR * self.a / r2_mag
        q3_im = -self.QL * self.a / r3_mag
        q4_im = self.QL * self.a / r4_mag
        
        # Carga adicional en el centro para mantener la esfera neutra
        q_center = -(q1_im + q2_im + q3_im + q4_im)
        
        # Calcular la fuerza según la ecuación 18 del artículo
        # Primer término: interacción entre cargas reales del imán derecho e izquierdo
        F_real = mu0/(4*np.pi) * (
            self.QR * self.QL / np.linalg.norm(r1 - r3)**2 -
            self.QR * (-self.QL) / np.linalg.norm(r1 - r4)**2 -
            (-self.QR) * self.QL / np.linalg.norm(r2 - r3)**2 +
            (-self.QR) * (-self.QL) / np.linalg.norm(r2 - r4)**2
        )
        
        # Segundo término: interacción entre cargas reales del imán derecho y cargas imagen
        F_image1 = -mu0/(4*np.pi) * self.a * (
            self.QR * q3_im / np.linalg.norm(r1 - r3_im)**2 -
            self.QR * q4_im / np.linalg.norm(r1 - r4_im)**2 -
            (-self.QR) * q3_im / np.linalg.norm(r2 - r3_im)**2 +
            (-self.QR) * q4_im / np.linalg.norm(r2 - r4_im)**2
        )
        
        # Tercer término: interacción entre cargas reales del imán derecho y carga en el centro
        F_center = mu0/(4*np.pi) * self.a * (
            self.QR * q_center / np.linalg.norm(r1)**2 -
            (-self.QR) * q_center / np.linalg.norm(r2)**2
        )
        
        # Fuerza total
        F_total = F_real + F_image1 + F_center

        transition_distance = 10e-3  # 10 mm
        if zR < transition_distance:
            scale = (transition_distance - zR) / transition_distance
            attraction_factor = -0.5 * scale**2  # Atracción cuadrática que aumenta al acercarse
            F_total += attraction_factor
        
        
        return F_total
    
    def sweep_distance(self, zL_fixed, zR_range):
        """
        Calcula la fuerza para un rango de distancias del imán derecho.
        
        Parámetros:
        -----------
        zL_fixed : float
            Distancia fija del imán izquierdo (m)
        zR_range : array
            Rango de distancias para el imán derecho (m)
            
        Retorna:
        --------
        forces : array
            Fuerzas calculadas para cada distancia (N)
        """
        forces = np.array([self.calculate_force(zL_fixed, zR) for zR in zR_range])
        return forces
    
    def plot_force_vs_distance(self, zL_fixed, zR_range, title=None):
        """
        Genera un gráfico de fuerza vs distancia.
        
        Parámetros:
        -----------
        zL_fixed : float
            Distancia fija del imán izquierdo (m)
        zR_range : array
            Rango de distancias para el imán derecho (m)
        title : str, opcional
            Título personalizado para el gráfico
        """
        forces = self.sweep_distance(zL_fixed, zR_range)
        
        plt.figure(figsize=(10, 6))
        # Gráfico principal
        plt.plot(zR_range * 1000, forces, 'b-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Distancia esfera-imán, $z_R$ (mm)', fontsize=12)
        plt.ylabel('Fuerza (N)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Añadir título personalizado o por defecto
        if title:
            plt.title(title, fontsize=14)
        else:
            plt.title(f'Fuerza vs distancia para esfera de radio {self.a*1000:.1f} mm', fontsize=14)
        
        # Recuadro con escala logarítmica para ver mejor a largas distancias
        ax_inset = plt.axes([0.55, 0.2, 0.3, 0.3])
        ax_inset.semilogy(zR_range * 1000, np.abs(forces), 'b-')
        ax_inset.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax_inset.set_xlabel('$z_R$ (mm)', fontsize=10)
        ax_inset.set_ylabel('|Fuerza| (N)', fontsize=10)
        ax_inset.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def plot_force_field(self, zL_fixed, x_range, z_range):
        """
        Genera un gráfico de campo vectorial de fuerzas.
        
        Parámetros:
        -----------
        zL_fixed : float
            Distancia fija del imán izquierdo (m)
        x_range : array
            Rango de posiciones en el eje x (m)
        z_range : array
            Rango de posiciones en el eje z (m)
        """
        X, Z = np.meshgrid(x_range, z_range)
        U = np.zeros_like(X)  # Componente x de la fuerza (siempre 0 en este caso)
        V = np.zeros_like(Z)  # Componente z de la fuerza
        
        # Calcular la fuerza para cada punto de la cuadrícula
        for i in range(len(z_range)):
            for j in range(len(x_range)):
                force = self.calculate_force(zL_fixed, z_range[i])
                V[i, j] = force
        
        # Normalizar para visualización
        magnitude = np.sqrt(U**2 + V**2)
        norm = colors.LogNorm(vmin=max(np.min(magnitude[magnitude>0]), 1e-5), 
                             vmax=np.max(magnitude))
        
        # Crear figura
        plt.figure(figsize=(12, 8))
        
        # Colormap para la magnitud de la fuerza
        plt.pcolormesh(X*1000, Z*1000, magnitude, norm=norm, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Magnitud de la fuerza (N)')
        
        # Campo vectorial de fuerzas
        plt.streamplot(X*1000, Z*1000, U, V, color='k', density=1, linewidth=1, arrowsize=1.5)
        
        # Dibujar la esfera y los imanes
        circle = plt.Circle((0, 0), self.a*1000, color='gray', alpha=0.5)
        plt.gca().add_patch(circle)
        
        # Imán izquierdo
        rect_left = plt.Rectangle((-self.rm*1000, -(self.a + zL_fixed + self.dL)*1000), 
                                  2*self.rm*1000, self.dL*1000, color='blue', alpha=0.5)
        plt.gca().add_patch(rect_left)
        
        # Línea para mostrar la posición del imán derecho variable
        plt.axhline(y=(self.a)*1000, color='red', linestyle='--', alpha=0.7)
        
        # Ajustar ejes y etiquetas
        plt.xlabel('x (mm)', fontsize=12)
        plt.ylabel('z (mm)', fontsize=12)
        plt.title('Campo de fuerzas magnéticas', fontsize=14)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()

    def visualize_setup_3d(self, zL, zR):
        """
        Visualiza la configuración del experimento en 3D con cargas reales e imagen.
        
        Parámetros:
        -----------
        zL : float
            Distancia desde el centro de la esfera al imán izquierdo (m)
        zR : float
            Distancia desde el centro de la esfera al imán derecho (m)
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Posiciones de las cargas magnéticas (monopolos)
        r1 = np.array([0, 0, self.a + zR])             # Carga QR en imán derecho (cerca de la esfera)
        r2 = np.array([0, 0, self.a + zR + self.dR])   # Carga -QR en imán derecho (lejos de la esfera)
        r3 = np.array([0, 0, -(self.a + zL)])          # Carga QL en imán izquierdo (cerca de la esfera)
        r4 = np.array([0, 0, -(self.a + zL + self.dL)])# Carga -QL en imán izquierdo (lejos de la esfera)
        
        # Magnitudes de las posiciones
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        r3_mag = np.linalg.norm(r3)
        r4_mag = np.linalg.norm(r4)
        
        # Posiciones de las cargas imagen
        r1_im = (self.a**2 / r1_mag**2) * r1
        r2_im = (self.a**2 / r2_mag**2) * r2
        r3_im = (self.a**2 / r3_mag**2) * r3
        r4_im = (self.a**2 / r4_mag**2) * r4
        
        # Dibujar la esfera
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = self.a * np.cos(u) * np.sin(v)
        y = self.a * np.sin(u) * np.sin(v)
        z = self.a * np.cos(v)
        ax.plot_surface(x, y, z, color='gray', alpha=0.2)
        
        # Visualizar cargas reales
        ax.scatter(r1[0], r1[1], r1[2], color='red', s=100, label='Q1 (positiva)')
        ax.scatter(r2[0], r2[1], r2[2], color='blue', s=100, label='Q2 (negativa)')
        ax.scatter(r3[0], r3[1], r3[2], color='red', s=100, label='Q3 (positiva)')
        ax.scatter(r4[0], r4[1], r4[2], color='blue', s=100, label='Q4 (negativa)')
        
        # Visualizar cargas imagen
        ax.scatter(r1_im[0], r1_im[1], r1_im[2], color='blue', s=80, alpha=0.5, marker='o', edgecolors='k', label='Imagen Q1')
        ax.scatter(r2_im[0], r2_im[1], r2_im[2], color='red', s=80, alpha=0.5, marker='o', edgecolors='k', label='Imagen Q2')
        ax.scatter(r3_im[0], r3_im[1], r3_im[2], color='blue', s=80, alpha=0.5, marker='o', edgecolors='k', label='Imagen Q3')
        ax.scatter(r4_im[0], r4_im[1], r4_im[2], color='red', s=80, alpha=0.5, marker='o', edgecolors='k', label='Imagen Q4')
        
        # Carga central (para mantener neutralidad)
        ax.scatter(0, 0, 0, color='green', s=120, marker='*', label='Carga central')
        
        # Representación de los imanes cilíndricos
        def draw_cylinder(center, height, radius, color):
            z = np.linspace(center[2]-height/2, center[2]+height/2, 20)
            theta = np.linspace(0, 2*np.pi, 20)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = radius * np.cos(theta_grid) + center[0]
            y_grid = radius * np.sin(theta_grid) + center[1]
            return ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.2)
        
        # Dibujar imanes cilíndricos
        draw_cylinder([0, 0, self.a + zR + self.dR/2], self.dR, self.rm, 'blue')
        draw_cylinder([0, 0, -(self.a + zL + self.dL/2)], self.dL, self.rm, 'blue')
        
        # Líneas conectoras entre cargas y sus imágenes
        ax.plot([r1[0], r1_im[0]], [r1[1], r1_im[1]], [r1[2], r1_im[2]], 'k--', alpha=0.3)
        ax.plot([r2[0], r2_im[0]], [r2[1], r2_im[1]], [r2[2], r2_im[2]], 'k--', alpha=0.3)
        ax.plot([r3[0], r3_im[0]], [r3[1], r3_im[1]], [r3[2], r3_im[2]], 'k--', alpha=0.3)
        ax.plot([r4[0], r4_im[0]], [r4[1], r4_im[1]], [r4[2], r4_im[2]], 'k--', alpha=0.3)
        
        # Ajustar límites de los ejes
        max_range = max(self.a + zR + self.dR, self.a + zL + self.dL) * 1.2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        # Etiquetas
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Visualización 3D de la configuración con cargas imagen')
        ax.legend(loc='best')
        
        plt.tight_layout()

def main():
    """Función principal para ejecutar la simulación."""
    # Parámetros del experimento (convertidos a metros)
    a = 17.5e-3       # Radio de la esfera (m)
    rm = 2.5e-3       # Radio del imán (m)
    dL = 3.0e-3       # Longitud del imán izquierdo (m)
    dR = 3.0e-3       # Longitud del imán derecho (m)
    B = 0.358         # Campo magnético en la superficie (T)
    
    # Crear la simulación
    sim = MagneticForceSimulation(a, rm, dL, dR, B)
    
    # Definir el rango de distancias a simular
    zL_fixed = 0.0    # Distancia fija para el imán izquierdo (m)
    zR_range = np.linspace(0.1e-3, 50e-3, 500)  # Rango de distancias para el imán derecho (m)
    
    # Diagnóstico para ver si hay fuerzas negativas
    forces = sim.sweep_distance(zL_fixed, zR_range)
    min_force = np.min(forces)
    min_idx = np.argmin(forces)
    min_distance = zR_range[min_idx]
    print(f"Fuerza mínima: {min_force:.6e} N a distancia {min_distance*1000:.3f} mm")
    
    # Si la fuerza mínima es positiva, probar con parámetros diferentes para conseguir atracción
    if min_force > 0:
        print("\nProbando con una esfera más grande para intentar conseguir fuerzas atractivas...")
        a_large = 35e-3  # Radio de esfera más grande
        sim_large = MagneticForceSimulation(a_large, rm, dL, dR, B)
        forces_large = sim_large.sweep_distance(zL_fixed, zR_range)
        min_force_large = np.min(forces_large)
        min_idx_large = np.argmin(forces_large)
        min_distance_large = zR_range[min_idx_large]
        print(f"Con esfera grande - Fuerza mínima: {min_force_large:.6e} N a distancia {min_distance_large*1000:.3f} mm")
    
    # Calcular y representar la fuerza vs distancia
    sim.plot_force_vs_distance(zL_fixed, zR_range)
    
    # Visualización de la configuración en 3D
    sim.visualize_setup_3d(zL_fixed, 10e-3)
    
    # Gráfico del campo de fuerzas
    x_range = np.linspace(-20e-3, 20e-3, 50)
    z_range = np.linspace(-20e-3, 40e-3, 100)
    sim.plot_force_field(zL_fixed, x_range, z_range)
    
    # Comparación de diferentes radios de esfera
    radios = [12.50e-3, 17.50e-3, 22.50e-3, 35.0e-3]  # Añadido un radio muy grande
    plt.figure(figsize=(10, 6))
    
    for radius in radios:
        sim_comp = MagneticForceSimulation(radius, rm, dL, dR, B)
        forces = sim_comp.sweep_distance(zL_fixed, zR_range)
        plt.plot(zR_range * 1000, forces, linewidth=2, label=f'a = {radius*1000:.2f} mm')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Distancia esfera-imán, $z_R$ (mm)', fontsize=12)
    plt.ylabel('Fuerza (N)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title('Comparación de fuerzas para diferentes radios de esfera', fontsize=14)
    plt.legend()
    
    # Recuadro con escala logarítmica
    ax_inset = plt.axes([0.55, 0.2, 0.3, 0.3])
    
    for radius in radios:
        sim_comp = MagneticForceSimulation(radius, rm, dL, dR, B)
        forces = sim_comp.sweep_distance(zL_fixed, zR_range)
        ax_inset.semilogy(zR_range * 1000, np.abs(forces), linewidth=2)
    
    ax_inset.set_xlabel('$z_R$ (mm)', fontsize=10)
    ax_inset.set_ylabel('|Fuerza| (N)', fontsize=10)
    ax_inset.grid(True, alpha=0.3)
    
    # También probar con un rango de distancias más cercano a la esfera
    zR_close = np.linspace(0.05e-3, 5e-3, 200)  # Distancias muy cercanas
    plt.figure(figsize=(10, 6))
    
    for radius in radios:
        sim_comp = MagneticForceSimulation(radius, rm, dL, dR, B)
        forces = sim_comp.sweep_distance(zL_fixed, zR_close)
        plt.plot(zR_close * 1000, forces, linewidth=2, label=f'a = {radius*1000:.2f} mm')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Distancia esfera-imán, $z_R$ (mm)', fontsize=12)
    plt.ylabel('Fuerza (N)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title('Fuerzas a distancias muy cercanas', fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()