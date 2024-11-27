% Generar datos aleatorios que sigan un plano con ruido
a_true = 2; % Coeficiente en x verdadero
b_true = 3; % Coeficiente en y verdadero
c_true = 5; % Término independiente verdadero
sigma_ruido = 1; % Desviación estándar del ruido
num_datos = 100; % Número de datos

x = linspace(-10, 10, num_datos)';
y = linspace(-10, 10, num_datos)';
[X, Y] = meshgrid(x, y);
Z_verdadero = a_true * X + b_true * Y + c_true;
Z_ruidoso = Z_verdadero + sigma_ruido * randn(size(X)); % Añadir ruido gaussiano

% Función objetivo para ajuste de un plano
plano = @(params) sum((params(1) * X + params(2) * Y + params(3) - Z_ruidoso).^2, 'all');

% Algoritmo CMA-ES
dimension = 3; % Número de parámetros: coeficiente en x, coeficiente en y y término independiente
sigma = 1;
tamano_poblacion = 100;
max_iter = 1000;
umbral_parada = 1e-5;

media = randn(1, dimension);
covarianza = eye(dimension) * sigma;
sigma = sigma / sqrt(dimension);

soluciones = [];
mejor_aptitud = inf;

for i = 1:max_iter
    poblacion = mvnrnd(media, covarianza, tamano_poblacion);
    aptitudes = zeros(1, tamano_poblacion);
    for j = 1:tamano_poblacion
        aptitudes(j) = plano(poblacion(j,:)); % Función objetivo para ajuste de un plano
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
    poblacion_ponderada = bsxfun(@minus, poblacion, media);
    media = media + (1 / tamano_poblacion) * sum(poblacion_ponderada);

    % Actualizar covarianza
    covarianza = (poblacion_ponderada' * poblacion_ponderada) / tamano_poblacion;
    covarianza = triu(covarianza) + triu(covarianza, 1)';  % Asegurar matriz simétrica

    % Actualizar sigma
    norma_media = norm(media);
    sigma = sigma * exp(0.05 * (norma_media / sqrt(dimension) - 1));
end

% Resultados
fprintf('Mejor Solución (a, b, c): (%f, %f, %f)\n', mejor_solucion(1), mejor_solucion(2), mejor_solucion(3));
fprintf('Mejor Aptitud: %f\n', mejor_aptitud);

% Visualización
figure;
scatter3(X(:), Y(:), Z_ruidoso(:), 10, 'b', 'filled'); hold on;
surf(X, Y, mejor_solucion(1) * X + mejor_solucion(2) * Y + mejor_solucion(3), 'FaceAlpha', 0.5);
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Ajuste de un plano usando CMA-ES');
legend('Datos ruidosos', 'Plano ajustado');
grid on;