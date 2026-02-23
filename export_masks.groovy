/**
 * QuPath Mask Exporter
 * 
 * Script para exportar máscaras de anotaciones como OME-TIFF piramidal.
 * Optimizado para imágenes de patología digital de gran tamaño (WSI).
 * 
 * Características:
 * - Escritura por tiles (no carga toda la imagen en memoria)
 * - Compresión LZW (lossless)
 * - Salida uint8 single-channel
 * - Pirámide multinivel (mismos niveles que export_images.groovy)
 * - Encaja 1:1 con las imágenes exportadas a cualquier resolución
 * 
 * Requisitos: QuPath 0.6.x o superior
 * 
 * Uso:
 * 1. Abrir proyecto en QuPath
 * 2. Modificar OUTPUT_DIR
 * 3. Ejecutar script (Ctrl+R)
 */

import qupath.lib.objects.PathAnnotationObject
import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.images.writers.ome.OMEPyramidWriter
import qupath.lib.common.GeneralTools

// =====================================================
// CONFIGURACIÓN
// =====================================================

// Directorio de salida (modificar según necesidad)
def OUTPUT_DIR = "/media/abel/TOSHIBA EXT/export/intento_final_tiffs"

// Tamaño de tile para escritura eficiente (512 es un buen balance)
def TILE_SIZE = 512

// Tamaño mínimo del nivel más bajo de la pirámide (en píxeles)
// IMPORTANTE: Usar el mismo valor que en export_images.groovy para que encajen
def MIN_PYRAMID_SIZE = 1024

// Crear directorio si no existe
mkdirs(OUTPUT_DIR)

// =====================================================
// FUNCIONES AUXILIARES
// =====================================================

/**
 * Calcula los niveles de downsample para la pirámide.
 * IMPORTANTE: Usa el mismo algoritmo que export_images.groovy
 * para que las máscaras y las imágenes encajen a cualquier resolución.
 * 
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

// =====================================================
// DEFINICIÓN DE CLASES
// El orden define el ID de cada clase (índice + 1)
// ID 0 = fondo (sin anotación)
// =====================================================

def CLASS_NAMES = [
    "Abnormal secretions",           // ID: 1
    "Adipose tissue",                // ID: 2
    "Artifact",                      // ID: 3
    "Atypical intraductal proliferation",  // ID: 4
    "Bening gland",                  // ID: 5
    "Blood vessels",                 // ID: 6
    "Fibromuscular bundles",         // ID: 7
    "High grade prostatic intraepithelial neoplasia (HGPIN)",  // ID: 8
    "Immune cells",                  // ID: 9
    "Intestinal glands and mucus",   // ID: 10
    "Intraductal carcinoma",         // ID: 11
    "Mitosis",                       // ID: 12
    "Muscle",                        // ID: 13
    "Necrosis",                      // ID: 14
    "Negative",                      // ID: 15
    "Nerve",                         // ID: 16
    "Nerve ganglion",                // ID: 17
    "Normal  secretions",            // ID: 18
    "Prominent nucleolus",           // ID: 19
    "Red blood cells",               // ID: 20
    "Seminal vesicle",               // ID: 21
    "Sin clasificación",             // ID: 22
    "Stroma",                        // ID: 23
    "Tumor"                          // ID: 24
]

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
print "QuPath Mask Exporter"
print "=========================================="
print "Proyecto: ${project.getName()}"
print "Imagenes: ${project.getImageList().size()}"
print "Clases definidas: ${CLASS_NAMES.size()}"
print "Directorio de salida: ${OUTPUT_DIR}"
print "Tamano de tile: ${TILE_SIZE}x${TILE_SIZE}"
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
    
    try {
        def imageData = entry.readImageData()
        def hierarchy = imageData.getHierarchy()
        def server = imageData.getServer()
        def imageName = GeneralTools.stripExtension(entry.getImageName())
        
        print "  Resolucion: ${server.getWidth()} x ${server.getHeight()}"
        
        // Obtener anotaciones
        def annotations = hierarchy.getObjects(null, PathAnnotationObject)
        
        if (annotations.isEmpty()) {
            print "  Sin anotaciones - omitiendo"
            skippedCount++
            return
        }
        
        // Contar clases presentes
        def classCounts = [:].withDefault { 0 }
        annotations.each {
            def pc = it.getPathClass()
            if (pc != null)
                classCounts[pc.getName()]++
        }
        
        print "  Anotaciones: ${annotations.size()}"
        print "  Clases presentes: ${classCounts.size()}"
        classCounts.each { k, v ->
            print "    - ${k}: ${v}"
        }
        
        // =====================================================
        // CREAR SERVIDOR DE ETIQUETAS
        // =====================================================
        
        def labelBuilder = new LabeledImageServer.Builder(imageData)
            .backgroundLabel(0)
            .useAnnotations()
            .multichannelOutput(false)
        
        CLASS_NAMES.eachWithIndex { name, i ->
            labelBuilder.addLabel(name, i + 1)
        }
        
        def labelServer = labelBuilder.build()
        
        // =====================================================
        // CALCULAR NIVELES DE PIRÁMIDE
        // (Mismos niveles que export_images.groovy)
        // =====================================================
        
        def downsamples = calculatePyramidLevels(server, MIN_PYRAMID_SIZE)
        print "  Niveles de piramide: ${downsamples.size()}"
        print "  Downsamples: ${downsamples.collect { String.format('%.0f', it) }.join(', ')}"
        
        // =====================================================
        // ESCRIBIR OME-TIFF PIRAMIDAL
        // =====================================================
        
        def outPath = buildFilePath(OUTPUT_DIR, "${imageName}__mask_multiclass.ome.tif")
        
        print "  Exportando..."
        def exportStart = System.currentTimeMillis()
        
        def writer = new OMEPyramidWriter.Builder(labelServer)
            .tileSize(TILE_SIZE)
            .downsamples(downsamples as double[])  // Pirámide multinivel
            .compression(OMEPyramidWriter.CompressionType.LZW)
            .parallelize()
            .build()
        
        writer.writeSeries(outPath)
        
        def exportTime = (System.currentTimeMillis() - exportStart) / 1000.0
        
        // Verificar resultado
        def outFile = new File(outPath)
        def fileSize = outFile.length()
        
        print String.format("  Guardado: %s (%s)", outFile.getName(), formatSize(fileSize))
        print String.format("  Tiempo: %.1f segundos", exportTime)
        
        totalBytesWritten += fileSize
        processedCount++
        
        // =====================================================
        // LIBERAR MEMORIA
        // =====================================================
        
        labelServer.close()
        imageData.getServer().close()
        System.gc()
        
    } catch (Exception e) {
        print "  ERROR: ${e.getMessage()}"
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
print "Omitidas (sin anotaciones): ${skippedCount}"
print "Errores: ${errorCount}"
print "Tamano total exportado: ${formatSize(totalBytesWritten)}"
print String.format("Tiempo total: %.1f segundos (%.1f minutos)", totalTime, totalTime / 60.0)
print "=========================================="
