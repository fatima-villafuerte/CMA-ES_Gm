G = load("G.txt");
z_ruidoso = load("Anomalia.txt");
z_ruidoso = z_ruidoso(:);
z_ruidoso2 = reshape(z_ruidoso, [25, 25]);
figure('Name','Modelo inicial','NumberTitle','off')
title('Modelo inicial')
mesh(z_ruidoso2)

% Método G*m
funcion_objetivo_Gm = @(params) sum((G * params - z_ruidoso) .^ 2);
m_invertido = G \ z_ruidoso;
residuo_Gm = funcion_objetivo_Gm(m_invertido);

% Resultados del método G*m
fprintf('Método G*m - Mejor Solución (a, b, c): (%f, %f, %f)\n', m_invertido(2), m_invertido(3), m_invertido(1));
fprintf('Método G*m - Residuo: %f\n', residuo_Gm);

% CMA-ES
dimension = length(m_invertido); % Número de parámetros: coeficientes en x, y y la ordenada al origen
sigma = 0.5;
tamano_poblacion = 100;
max_iter = 1000;
umbral_parada = 1e-10;
soluciones=[];

media = m_invertido'; % Inicializamos en m_invertido'
covarianza = eye(dimension) * sigma;

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
    
    % Ajustar sigma
    sigma = sigma * exp(0.05 * (norm(media) / sqrt(dimension) - 1));
end

% Resultados del método CMA-ES
fprintf('CMA-ES - Mejor Solución (a, b, c): (%f, %f, %f)\n', mejor_solucion(2), mejor_solucion(3), mejor_solucion(1));
fprintf('CMA-ES - Mejor Aptitud: %f\n', mejor_aptitud);

% Visualización
zf = G * mejor_solucion(:);
figure('Name','Modelo invertido','NumberTitle','off')
title('Modelo invertido')
zf = reshape(zf, [25, 25]);
mesh(zf)
e = zf(:) - z_ruidoso;
E = e' * e;