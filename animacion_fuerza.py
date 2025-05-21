#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animación de la atracción entre imanes con polos iguales mediada por una esfera ferromagnética.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simulacion_atraccion_magnetica import MagneticForceSimulation

class ForceAnimation:
    def __init__(self, sim, zL_fixed=0.0, zR_range=None):
        """
        Inicializa la animación.
        
        Parámetros:
        -----------
        sim : MagneticForceSimulation
            Objeto de simulación configurado
        zL_fixed : float
            Distancia fija del imán izquierdo (m)
        zR_range : array, opcional
            Rango de distancias para el imán derecho (m)
        """
        self.sim = sim
        self.zL_fixed = zL_fixed
        
        if zR_range is None:
            self.zR_range = np.linspace(0.5e-3, 40e-3, 100)
        else:
            self.zR_range = zR_range
        
        # Calcular fuerzas para todo el rango
        self.forces = sim.sweep_distance(zL_fixed, self.zR_range)
        
        # Configurar la figura
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.suptitle('Atracción entre imanes con polos iguales mediada por una esfera ferromagnética', 
                          fontsize=14)
        
        # Inicializar el gráfico de fuerza vs distancia
        self.line, = self.ax1.plot(self.zR_range * 1000, self.forces, 'b-', linewidth=2)
        self.point, = self.ax1.plot([], [], 'ro', markersize=8)
        
        self.ax1.set_xlabel('Distancia esfera-imán, $z_R$ (mm)', fontsize=12)
        self.ax1.set_ylabel('Fuerza (N)', fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Inicializar variables para los objetos que se actualizarán
        self.magnet_right = None
        self.force_arrow = None
        self.force_text = None
        
        # Inicializar la visualización esquemática
        self.setup_schematic_view()
        
    def setup_schematic_view(self):
        """Configura la vista esquemática de la configuración experimental."""
        self.ax2.set_xlim(-30, 30)
        self.ax2.set_ylim(-30, 50)
        self.ax2.set_aspect('equal')
        self.ax2.set_xlabel('X (mm)', fontsize=12)
        self.ax2.set_ylabel('Z (mm)', fontsize=12)
        self.ax2.grid(True, alpha=0.3)
        
        # Dibujar la esfera
        circle = plt.Circle((0, 0), self.sim.a * 1000, color='gray', alpha=0.7)
        self.ax2.add_patch(circle)
        
        # Dibujar el imán izquierdo (fijo)
        left_magnet = plt.Rectangle(
            (-self.sim.rm * 1000, -(self.sim.a + self.zL_fixed + self.sim.dL) * 1000),
            2 * self.sim.rm * 1000, self.sim.dL * 1000,
            color='blue', alpha=0.7
        )
        self.ax2.add_patch(left_magnet)
        self.ax2.text(-20, -(self.sim.a + self.zL_fixed + self.sim.dL/2) * 1000, 
                     'S', color='white', ha='center', va='center', fontsize=12)
        
        # Crear el imán derecho (se moverá durante la animación)
        rect = plt.Rectangle(
            (-self.sim.rm * 1000, (self.sim.a + self.zR_range[0]) * 1000),
            2 * self.sim.rm * 1000, self.sim.dR * 1000,
            color='blue', alpha=0.7
        )
        self.ax2.add_patch(rect)
        self.magnet_right = rect  # Guardar referencia
        
        # Añadir etiqueta 'S'
        self.ax2.text(-20, (self.sim.a + self.zR_range[0] + self.sim.dR/2) * 1000,
                     'S', color='white', ha='center', va='center', fontsize=12)
        
        # Flecha que muestra la dirección de la fuerza
        force_val = self.forces[0]
        arrow_length = min(max(5, abs(force_val) * 20), 40) * np.sign(force_val)
        
        arrow = self.ax2.arrow(
            15, (self.sim.a + self.zR_range[0] + self.sim.dR/2) * 1000,
            0, arrow_length,
            head_width=3, head_length=5, fc='red', ec='red', linewidth=2
        )
        self.force_arrow = arrow  # Guardar referencia
        
        # Texto para mostrar el valor de la fuerza
        text = self.ax2.text(
            20, (self.sim.a + self.zR_range[0] + self.sim.dR/2) * 1000,
            f'F = {force_val:.4f} N',
            color='red', ha='left', va='center', fontsize=10
        )
        self.force_text = text  # Guardar referencia
        
    def update(self, frame):
        """Actualiza la animación para el cuadro dado."""
        # Índice actual en el rango de distancias
        idx = frame % len(self.zR_range)
        zR = self.zR_range[idx]
        force = self.forces[idx]
        
        # Actualizar el punto en el gráfico de fuerza vs distancia
        self.point.set_data([zR * 1000], [force])
        
        # Actualizar la posición del imán derecho
        self.magnet_right.set_y((self.sim.a + zR) * 1000)
        
        # Actualizar la posición del texto 'S' en el imán derecho
        for text in self.ax2.texts:
            if text.get_position()[0] == -20 and text.get_position()[1] > 0:
                text.set_position((-20, (self.sim.a + zR + self.sim.dR/2) * 1000))
        
        # Eliminar la flecha anterior y crear una nueva
        if self.force_arrow:
            self.force_arrow.remove()
        
        # Crear una nueva flecha
        arrow_length = min(max(5, abs(force) * 20), 40) * np.sign(force)
        color = 'blue' if force > 0 else 'red'
        
        self.force_arrow = self.ax2.arrow(
            15, (self.sim.a + zR + self.sim.dR/2) * 1000,
            0, arrow_length,
            head_width=3, head_length=5, fc=color, ec=color, linewidth=2
        )
        
        # Actualizar el texto de la fuerza
        self.force_text.set_position((20, (self.sim.a + zR + self.sim.dR/2) * 1000))
        self.force_text.set_text(f'F = {force:.4f} N')
        self.force_text.set_color(color)
        
        return self.point, self.magnet_right, self.force_arrow, self.force_text
    
    def create_animation(self, frames=200, interval=50):
        """
        Crea la animación.
        
        Parámetros:
        -----------
        frames : int
            Número de cuadros en la animación
        interval : int
            Intervalo entre cuadros en milisegundos
            
        Retorna:
        --------
        anim : FuncAnimation
            Objeto de animación
        """
        return FuncAnimation(
            self.fig, self.update, frames=frames,
            interval=interval, blit=False  # Cambiado a False para evitar problemas
        )

def main():
    """Función principal para crear y mostrar la animación."""
    # Parámetros del experimento (convertidos a metros)
    a = 17.5e-3       # Radio de la esfera (m)
    rm = 2.5e-3       # Radio del imán (m)
    dL = 3.0e-3       # Longitud del imán izquierdo (m)
    dR = 3.0e-3       # Longitud del imán derecho (m)
    B = 0.358         # Campo magnético en la superficie (T)
    
    # Crear la simulación
    sim = MagneticForceSimulation(a, rm, dL, dR, B)
    
    # Definir rango de distancias para la animación
    # Nos enfocamos más en la región donde la fuerza cambia de signo
    zR_animation = np.concatenate([
    np.linspace(30e-3, 5e-3, 50),
    np.linspace(5e-3, 0.2e-3, 100),
    np.linspace(0.2e-3, 0.05e-3, 20)
    ])
    
    # Crear y mostrar la animación
    animation = ForceAnimation(sim, zL_fixed=0.0, zR_range=zR_animation)
    anim = animation.create_animation(frames=len(zR_animation), interval=50)
    
    # Guardar la animación (opcional, requiere ffmpeg o imagemagick)
    # anim.save('atraccion_magnetica.gif', writer='imagemagick', fps=20)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()