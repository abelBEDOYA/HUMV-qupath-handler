/**
 * QuPath Image Exporter (MRXS → OME-TIFF)
 * 
 * Script para exportar imágenes del proyecto a formato OME-TIFF piramidal.
 * Optimizado para imágenes de patología digital de gran tamaño (WSI).
 * 
 * Características:
 * - Escritura por tiles (no carga toda la imagen en memoria)
 * - Compresión LZW (lossless)
 * - Pirámide multinivel para visualización eficiente
 * - Paralelización para aprovechar múltiples cores
 * 
 * Requisitos: QuPath 0.6.x o superior
 * 
 * Uso:
 * 1. Abrir proyecto en QuPath
 * 2. Modificar OUTPUT_DIR
 * 3. Ejecutar script (Ctrl+R)
 * 
 * ADVERTENCIA: Las imágenes de salida pueden ocupar varios GB cada una.
 * Asegúrate de tener suficiente espacio en disco.
 */

import qupath.lib.images.writers.ome.OMEPyramidWriter
import qupath.lib.common.GeneralTools

// =====================================================
// CONFIGURACIÓN
// =====================================================

// Directorio de salida (modificar según necesidad)
def OUTPUT_DIR = "/media/abel/TOSHIBA EXT/export/images_tiff"

// Tamaño de tile para escritura eficiente
def TILE_SIZE = 512

// Tamaño mínimo del nivel más bajo de la pirámide (en píxeles)
def MIN_PYRAMID_SIZE = 1024

// Tipo de compresión: LZW (lossless), JPEG (lossy), ZLIB, UNCOMPRESSED
def COMPRESSION = OMEPyramidWriter.CompressionType.LZW

// Crear directorio si no existe
mkdirs(OUTPUT_DIR)

// =====================================================
// FUNCIONES AUXILIARES
// =====================================================

/**
 * Calcula los niveles de downsample para la pirámide.
 * Genera niveles 1, 4, 16, 64, 256... hasta que el tamaño sea < MIN_SIZE
 */
def calculatePyramidLevels(server, minSize) {
    def width = server.getWidth()
    def height = server.getHeight()
    def maxDim = Math.max(width, height)
    
    def downsamples = [1.0d]
    def ds = 4.0d
    
    while (maxDim / ds > minSize) {
        downsamples.add(ds)
        ds *= 4.0d
    }
    
    // Añadir un último nivel para thumbnail
    if (maxDim / ds > 256) {
        downsamples.add(ds)
    }
    
    return downsamples
}

/**
 * Formatea el tamaño en bytes a una cadena legible
 */
def formatSize(bytes) {
    if (bytes < 1024) return "${bytes} B"
    if (bytes < 1024 * 1024) return String.format("%.2f KB", bytes / 1024.0)
    if (bytes < 1024 * 1024 * 1024) return String.format("%.2f MB", bytes / (1024.0 * 1024.0))
    return String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0))
}

/**
 * Estima el tamaño de salida (aproximado)
 */
def estimateOutputSize(server, downsamples) {
    def width = server.getWidth()
    def height = server.getHeight()
    def channels = server.nChannels()
    def bytesPerPixel = channels * (server.getPixelType().getBitsPerPixel() / 8)
    
    def totalPixels = 0L
    downsamples.each { ds ->
        def w = (long)(width / ds)
        def h = (long)(height / ds)
        totalPixels += w * h
    }
    
    // Estimación con compresión LZW (~30-50% del original para imágenes típicas)
    def uncompressedSize = totalPixels * bytesPerPixel
    def estimatedCompressed = (long)(uncompressedSize * 0.4)
    
    return [uncompressed: uncompressedSize, estimated: estimatedCompressed]
}

// =====================================================
// VERIFICAR PROYECTO
// =====================================================

def project = getProject()
if (project == null) {
    print "ERROR: No hay proyecto abierto"
    print "Por favor, abre un proyecto QuPath antes de ejecutar este script"
    return
}

print "=========================================="
print "QuPath Image Exporter (MRXS -> OME-TIFF)"
print "=========================================="
print "Proyecto: ${project.getName()}"
print "Imagenes: ${project.getImageList().size()}"
print "Directorio de salida: ${OUTPUT_DIR}"
print "Tamano de tile: ${TILE_SIZE}x${TILE_SIZE}"
print "Compresion: ${COMPRESSION}"
print "=========================================="
print ""

// =====================================================
// PROCESAR CADA IMAGEN DEL PROYECTO
// =====================================================

def processedCount = 0
def skippedCount = 0
def errorCount = 0
def totalBytesWritten = 0L

def startTime = System.currentTimeMillis()

project.getImageList().each { entry ->
    
    print "--------------------------------------------"
    print "Procesando: ${entry.getImageName()}"
    
    // Verificar si ya existe
    def imageName = GeneralTools.stripExtension(entry.getImageName())
    def outPath = buildFilePath(OUTPUT_DIR, "${imageName}.ome.tif")
    def outFile = new File(outPath)
    
    if (outFile.exists()) {
        print "  Ya existe - omitiendo"
        print "  (Elimina el archivo si quieres regenerarlo)"
        skippedCount++
        return
    }
    
    try {
        def imageData = entry.readImageData()
        def server = imageData.getServer()
        
        def width = server.getWidth()
        def height = server.getHeight()
        def channels = server.nChannels()
        def pixelType = server.getPixelType()
        
        print "  Resolucion: ${width} x ${height}"
        print "  Canales: ${channels}"
        print "  Tipo de pixel: ${pixelType}"
        
        // Calcular niveles de pirámide
        def downsamples = calculatePyramidLevels(server, MIN_PYRAMID_SIZE)
        print "  Niveles de piramide: ${downsamples.size()}"
        print "  Downsamples: ${downsamples.collect { String.format('%.0f', it) }.join(', ')}"
        
        // Estimar tamaño
        def sizeEstimate = estimateOutputSize(server, downsamples)
        print "  Tamano estimado: ${formatSize(sizeEstimate.estimated)} (sin comprimir: ${formatSize(sizeEstimate.uncompressed)})"
        
        // =====================================================
        // EXPORTAR IMAGEN
        // =====================================================
        
        print "  Exportando..."
        def exportStart = System.currentTimeMillis()
        
        def writer = new OMEPyramidWriter.Builder(server)
            .tileSize(TILE_SIZE)
            .downsamples(downsamples as double[])
            .compression(COMPRESSION)
            .parallelize()
            .build()
        
        writer.writeSeries(outPath)
        
        def exportTime = (System.currentTimeMillis() - exportStart) / 1000.0
        
        // Verificar resultado
        outFile = new File(outPath)
        def fileSize = outFile.length()
        totalBytesWritten += fileSize
        
        print String.format("  Guardado: %s (%s)", outFile.getName(), formatSize(fileSize))
        print String.format("  Tiempo: %.1f segundos", exportTime)
        
        processedCount++
        
        // =====================================================
        // LIBERAR MEMORIA
        // =====================================================
        
        server.close()
        imageData.getServer().close()
        System.gc()
        
        // Pequeña pausa para estabilizar
        Thread.sleep(500)
        
    } catch (Exception e) {
        print "  ERROR: ${e.getMessage()}"
        e.printStackTrace()
        errorCount++
    }
}

// =====================================================
// RESUMEN FINAL
// =====================================================

def totalTime = (System.currentTimeMillis() - startTime) / 1000.0

print ""
print "=========================================="
print "PROCESO COMPLETADO"
print "=========================================="
print "Procesadas: ${processedCount}"
print "Omitidas (ya existian): ${skippedCount}"
print "Errores: ${errorCount}"
print "Tamano total exportado: ${formatSize(totalBytesWritten)}"
print String.format("Tiempo total: %.1f segundos (%.1f minutos)", totalTime, totalTime / 60.0)
print "=========================================="
