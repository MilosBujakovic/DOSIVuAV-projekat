import cv2 ### LIBRARY USED: UOpenCV 1.0.0
import numpy as np


def warper(img, src, dst):
    #Compute an apply perspective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST) #keep same size as input image

    return warped

def enhance_contrast(grayscale_image):
    # CLAHE metoda sluzi za prilagodjeno histogramsko izjednacavanje (AHE), tj. pomjesno izjednacava boje u izdvojenim
    # dijelovima slike (tileGrid) cime ne samo da gubimo razlike kod manjih sumova nego postizemo i povecanje kontrasta
    # izmedju onih dijelova koji su vise razliciti, medjutim kako oni dijelovi na kojima je promjena najveca ne bi u
    # potpunosti preuzeli primat i "zaslijepili" nas za ostale, neophodno je ograniciti nivo kontrasta sa clipLimit
    # parametrom koji nacesce uzima vrijednosti od 1 do 4 otkuda i naziv CL-AHE, CLAHE nam omogucava lakse uociti trake
    # na autoputu u uslovima vecih promjena okruzenja (npr. zakrpa svjetlije boje ili sijenke drveca u challenge01.mp4)
    # vrlo lako im se prilagodjava i jasno uocava i trake razlicitih boja (u nasem primjeru zute i bijele)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(grayscale_image)

    # korekcija gamme preko lookup tabele sluzi izjednacavanju osvjetljenja izmedju svjetlije i tamnije prepoznatih
    # dijelova trake (npr. onih koji su na suncu i onih u sijenci drveta) cv2.LUT zapravo preslikava sve vrijednosti iz
    # lookup-tabele na vrijednosti koje se nalaze u nasoj slici na vrlo brz nacin jer vrsi prostu zamjenu svakog polja
    # sa onim na kojem se nalazi preporucena vrijednost u tabeli, kako je nasa alpha < 1.0, to znaci da zelimo posvijetliti
    # one dijelove koji su tamniji i time omoguciti da ne ispadnu iz praga ogranicenja kada budemo pravili binarnu sliku
    # ovime smanjujemo isprekidanost pri otkrivanju traka tamo gdje ono zapravo ne postoji nego je uzrokovano losim osvjetljenjem
    table = np.array([(i / 255.0) ** 0.8 * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(enhanced_gray, table)

def lanes_detection(binary_image):
    histogram = np.sum(binary_image[binary_image.shape[0] // 2:, :], axis=0)
    midpoint = np.int32(histogram.shape[0] / 2)
    leftTrack_base = np.argmax(histogram[:midpoint])
    rightTrack_base = np.argmax(histogram[midpoint:]) + midpoint

    # otkrivamo trake od dna (ispred) ka vrhu (u daljini) slike
    y = 530
    left_centersX = []
    right_centersX = []
    lane_detection = binary_image.copy()
    while y > 0:
        # otkrivanje lijeve trake
        final_image = binary_image[y - 60:y, leftTrack_base - 40:leftTrack_base + 40]
        contours, _ = cv2.findContours(final_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                leftTrack_base = leftTrack_base - 40 + cx
                left_centersX.append(leftTrack_base)

        # otkrivanje desne trake
        final_image = binary_image[y - 60:y, rightTrack_base - 40:rightTrack_base + 40]
        contours, _ = cv2.findContours(final_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                rightTrack_base = rightTrack_base - 40 + cx
                right_centersX.append(rightTrack_base)

        cv2.rectangle(lane_detection, pt1=(leftTrack_base - 40, y),
                      pt2=(leftTrack_base + 40, y - 60), color=(255, 255, 255), thickness=3)
        cv2.rectangle(lane_detection, pt1=(rightTrack_base - 40, y),
                      pt2=(rightTrack_base + 40, y - 60), color=(255, 255, 255), thickness=3)
        y -= 60

    #izdvajamo samo pixele na kojima smo prepoznali trake kolosijeka
    lanes_coordinates = binary_image.nonzero()
    lefty = lanes_coordinates[0][left_centersX]
    leftx = lanes_coordinates[1][left_centersX]

    righty = lanes_coordinates[0][right_centersX]
    rightx = lanes_coordinates[1][right_centersX]

    #nasa konacna procjena (aproksimacija) izgleda traka
    left_centers = np.polyfit(lefty, leftx, 2)
    right_centers = np.polyfit(righty, rightx, 2)

    return lane_detection, left_centers, right_centers

def position_and_curvature_estimation(binary_warped, left_centers, right_centers):
    # odrjedjujemo koliku duzinu predstavljaju pixeli (sracunatu na kraju lanes_detection)
    # pomocu tackica i zvanicnih podataka o nacinu pravljenja puteva
    yPixel_size = 20 / 540  # velicina pixela po y osi u metrima
    xPixel_size = 3.66 / 675  # velicina pixela po x osi u metrima

    image_height = binary_warped.shape[0]
    image_width = binary_warped.shape[1]

    # izracunavanje trenutnog polozaja kola na kolosijeku u odnosu na njeno srediste
    car_center = image_width / 2  # srediste slike = srediste kola

    #procjena polozaja lijeve i desne linije trake na slici po prepoznavanju sa binary slike
    left_track_bottom = left_centers[0] * (image_height - 1) ** 2 + left_centers[1] * (image_height - 1) + left_centers[2]
    right_track_bottom = right_centers[0] * (image_height - 1) ** 2 + right_centers[1] * (image_height - 1) + right_centers[2]
    lane_center = (left_track_bottom + right_track_bottom) / 2 #procjena sredista vozne trake u kojoj vozimo

    #procjena polozaja kola u odnosu na srediste voznog kolosijeka izrazena u metrima
    # <0 znaci lijevo od sredista, >0 znaci desno od sredista
    vehicle_position = (car_center - lane_center) * xPixel_size


    #izracunavanje stepena zakrivljenja traka na putu za prikaz na ekranu
    # pravimo ravnomjerne razmake po cijeloj Y osi za iscrtavanje
    y_pixels = np.linspace(0, image_height - 1, image_height)

    # procjena (aproksimacija) oblika traka polinomom za izracunavanje zakrivljenja
    left_track_curve = np.polyfit(y_pixels * yPixel_size,left_centers[0] * y_pixels ** 2 * xPixel_size +
                                  left_centers[1] * y_pixels * xPixel_size + left_centers[2] * xPixel_size, 2)
    right_track_curve = np.polyfit(y_pixels * yPixel_size, right_centers[0] * y_pixels ** 2 * xPixel_size +
                                  right_centers[1] * y_pixels * xPixel_size + right_centers[2] * xPixel_size, 2)
    #y_eval = np.max(y_pixels)
    # TODO: izracunati ugao-radijus zakrivljenja

    return vehicle_position, left_track_bottom, right_track_bottom

def highlight_lanes_data(image, binary_warped_image, left_centers, right_centers, image_points, points_of_interest):
    #pravljenje maske na koju cemo iscrtati pronadjenu liniju
    image_height = binary_warped_image.shape[0]
    warped_lane = np.full_like(image, 0)

    #izracunavanje X koordinate za linije u traci
    y_pixels = np.linspace(0, image_height - 1, image_height)
    left_trackX = left_centers[0] * y_pixels ** 2 + left_centers[1] * y_pixels + left_centers[2]
    right_trackX = right_centers[0] * y_pixels ** 2 + right_centers[1] * y_pixels + right_centers[2]

    #smijestanje svega u tacke
    pts_left = np.array([np.transpose(np.vstack([left_trackX, y_pixels]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_trackX, y_pixels])))])
    pts = np.hstack((pts_left, pts_right))

    #procjena (aproksimacija) povrsine koja predstavlja kolosijek u kojem se krecemo
    inverted_warp_perspective = cv2.getPerspectiveTransform(points_of_interest, image_points)
    cv2.fillPoly(warped_lane, np.int_([pts]), (255, 0, 0))

    #transformacija i mapiranje aproksimacije na izvornu sliku radi prikaza rezultata
    rewarped_detected_lane = cv2.warpPerspective(warped_lane, inverted_warp_perspective, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(image, 1, rewarped_detected_lane, 0.5, 0)

    return result

def main():
    #TODO: ovdje mozemo mijenjati snimke koje zelimo ispitivati
    cap = cv2.VideoCapture('./../test_videos/project_video01.mp4')

    #zapis snimka nakon analize
    #lanes_detection_out = cv2.VideoWriter('./../outputs/lanes_detection.avi', -1, 25.0, (960, 540))
    #final_out = cv2.VideoWriter('./../outputs/output.avi', -1, 25.0, (960, 540))

    #odabir tacaka koje ce nam ogranicavati prostor bitan za ispitivanje (u ovom slucaju autoput)
    #nepromjenjiv je, te ne mora biti u petlji
    topR_track = [590, 350]
    bottomR_track = [960, 500]
    bottomL_track = [150, 500]
    topL_track = [403, 350]
    points_of_interest = np.float32([topL_track, bottomL_track, topR_track, bottomR_track])
    image_points = np.float32([[0, 0], [0, 540], [960, 0], [960, 540]])

    #while petlju koristimo kako bismo nastavili izvrsavanje aplikacije za svaku sliku koja nam pristize sa video snimka
    while cap.isOpened():
        ret, img = cap.read()
        if not ret: break #exit flag
        image = cv2.resize(img, [960, 540]) #mijenjanje velicine kako bismo lakse poredili sve snimke
        #img_height = img.shape[0]
        #img_width = img.shape[1]

        #prikaz radi provjere polozaja tacaka u odnosu na kola, autoput u izvornoj slici
        cv2.circle(image, topL_track, 5, (0,255,0), -1)
        cv2.circle(image, topR_track, 5, (0, 255, 0), -1)
        cv2.circle(image, bottomR_track, 5, (0, 255, 0), -1)
        cv2.circle(image, bottomL_track, 5, (0, 255, 0), -1)
        cv2.imshow("original", image)

        #nakon ustanovljavanja dobrog izbora tacaka
        warped_image = warper(image, points_of_interest, image_points)
        cv2.imshow("warped image", warped_image)

        # filtriranje binarne slike - radi ispravljanja
        grayscale_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.medianBlur(grayscale_warped, 5)

        # segmentacija slike u binarnu
        gamma_corrected = enhance_contrast(grayscale)
        _, binary_image = cv2.threshold(gamma_corrected, 215, 255, cv2.THRESH_BINARY)
        cv2.imshow("threshold segmented", binary_image)

        # algoritam za otkrivanje linija na binarnoj slici
        lane_detection, left_centers, right_centers = lanes_detection(binary_image)

        #mjerenje rastojanja na snimcima: prosjecna sirina trake u Californiji 12ft = 3,6576m
        #rucno ustanovljavanje razdaljine u pixel-ima po x radi utvrdjivanja rastojanja
        #cv2.circle(lane_detection, (100,540),5,(255,0,0), 2)
        #cv2.circle(lane_detection, (775, 540), 5, (255, 0, 0), 2)
        # duzina isprekidane linije 10ft 30ft razmak izmedju => po duzini priblizno prikazano 60+ft ~= 20m
        # rucno ustanovljavanje razdaljine u pixel-ima po y radi utvrdjivanja rastojanja
        cv2.imshow("Lane detection test", lane_detection)

        #procjena polozaja kola u odnosu na srediste kolosijeka
        vehicle_position, left_track_bottom, right_track_bottom = position_and_curvature_estimation(binary_image, left_centers, right_centers)

        #points_of_interest i image_points zamjenjuju mjesta, jer sada zelimo
        # nasu transformisanu sliku ceste namapirati na izvorni snimak

        output_frame = highlight_lanes_data(image, binary_image, left_centers, right_centers,
                                            image_points=points_of_interest, points_of_interest=image_points)

        #dodavanje pozicije na izlaz
        side = "center"
        if vehicle_position < 0:
            side = "left"
        elif vehicle_position > 0:
            side = "right"
        if side!= "center":
            position =f"Vehicle position: {abs(vehicle_position):.1f} cm {side} from center"
        else:
            position = "Vehicle position is at the center of the lane"

        cv2.putText(output_frame,position,(50,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow("Final output",output_frame)

        #ispis slika u video datoteke
        #lanes_detection_out.write(lane_detection)
        #final_out.write(output_frame)

        if cv2.waitKey(25) & 0xFF == 27: break

    #final_out.release()
    #lanes_detection_out.release()
    cap.release()
    cv2.destroyAllWindows()

main()


def test_blur():
    img_src_path = "./../test_images/challange00101.jpg"
    image = cv2.imread(img_src_path)
    # after testing
    gaussian = cv2.GaussianBlur(image, (9,9), 0)
    median7 = cv2.medianBlur(image, 7)
    median9 = cv2.medianBlur(image, 9)
    median11 = cv2.medianBlur(image, 11)
    dst = cv2.fastNlMeansDenoisingColored(image, 6, 6, 7, 21)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    cv2.imshow("Median9 blur", median9)
    cv2.imshow("Median7 blur", median7)
    #cv2.imshow("Gussian blur", gaussian)
    cv2.imshow("Median11 blur", median11)
    #cv2.imshow("fastNlMeans ", dst)

    cv2.waitKey()