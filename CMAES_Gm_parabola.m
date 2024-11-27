% Generar datos aleatorios que sigan una parábola con ruido
a_true = 2; % Coeficiente cuadrático verdadero
b_true = 5; % Coeficiente lineal verdadero
c_true = 7; % Intersección en y verdadera
sigma_ruido = 1; % Desviación estándar del ruido
num_datos = 100; % Número de datos

x = linspace(-5, 5, num_datos)';
G = [ones(num_datos, 1) x x.^2];
m = [c_true; b_true; a_true];
y_verdadero = G * m;
y_ruidoso = y_verdadero + sigma_ruido * randn(size(x));

% Método G*m
funcion_objetivo_Gm = @(params) sum((G * params - y_ruidoso) .^ 2);
m_invertido = G \ y_ruidoso;
residuo_Gm = funcion_objetivo_Gm(m_invertido);

% Resultados del método G*m
fprintf('Método G*m - Mejor Solución (a, b, c): (%f, %f, %f)\n', m_invertido(3), m_invertido(2), m_invertido(1));
fprintf('Método G*m - Residuo: %f\n', residuo_Gm);

% CMA-ES
dimension = 3; % Número de parámetros: coeficientes cuadrático, lineal y la ordenada al origen
sigma = 0.5;
tamano_poblacion = 500;
max_iter = 1000;
umbral_parada = 1e-10;

media = randn(1, dimension); % Podríamos inicializar en m_invertido'
covarianza = eye(dimension) * sigma;

soluciones = [];
mejor_aptitud = inf;

for i = 1:max_iter
    poblacion = mvnrnd(media, covarianza, tamano_poblacion);
    aptitudes = zeros(1, tamano_poblacion);
    for j = 1:tamano_poblacion
        aptitudes(j) = funcion_objetivo_Gm(poblacion(j, :)');
    end
    
    [aptitudes, indices_ordenados] = sort(aptitudes);
    poblacion = poblacion(indices_ordenados, :);
    
    if aptitudes(1) < mejor_aptitud
        mejor_solucion = poblacion(1, :);
        mejor_aptitud = aptitudes(1);
    end
    
    % Guardamos la solución para visualizar la trayectoria
    soluciones = [soluciones; mejor_solucion];
    
    if max(abs(poblacion - media)) < umbral_parada
        break;
    end
    
    % Actualizar media
    media = mean(poblacion);
    
    % Actualizar covarianza
    covarianza = cov(poblacion);
    
    % Actualizar sigma
    sigma = sigma * exp(0.05 * (norm(media) / sqrt(dimension) - 1));
end

% Resultados del método CMA-ES
fprintf('CMA-ES - Mejor Solución (a, b, c): (%f, %f, %f)\n', mejor_solucion(3), mejor_solucion(2), mejor_solucion(1));
fprintf('CMA-ES - Mejor Aptitud: %f\n', mejor_aptitud);

% Visualización
figure;
plot(x, y_ruidoso, 'b.', 'MarkerSize', 10); hold on;
plot(x, G * m_invertido, 'g-', 'LineWidth', 2); % Línea ajustada por G*m
plot(x, G * mejor_solucion', 'r-', 'LineWidth', 2); % Línea ajustada por CMA-ES
xlabel('X');
ylabel('Y');
title('Ajuste de una parábola usando el método de G*m y CMA-ES');
legend('Datos ruidosos', 'Ajuste G*m', 'Ajuste CMA-ES');
grid on;