## Writeup Template

### You use this file as a template for your writeup.

Милош Бујаковић Е2-157/2024
---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/odabir_tacaka.png "Undistorted"
[image2]: ./examples/kalibracija_po_tackama.png "Road Transformed"
[image3]: ./examples/binary_slika.png "Binary Example"
[image4]: ./examples/preoblicenje_slike.png "Warp Example"
[image5]: ./examples/prepoznavanje_traka.png "Fit Visual"
[image6]: ./examples/krajnji_izlaz.png "Output"

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Потребно је исправити слику правилним одабиром координата као и њиховом пребацивањем и преобличавањем слике које је подробније описано у тачки 3, а суштина је добити "испеглану" слику на којој ћемо лакше пронаћи тачке од значаја и самим тиме прецизније одриједити податке који су нам битни и издвојити их од шумова тј. небитних. Примјер слике испод:

[image1] [image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Промјене боја у нијансе сиве су коришћене ради смањења количине обраде, јер тада нам не треба палета 3 боје/ствари него само 1 са којом успјешно чувамо оно што нам је кључно за распознавање на слици, ово је извршено методом cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY) у линији 185, након тога трансформације градијената и боја су настављене појачањем контраста између линија трака и остатка пута (како бисмо омогућили успјешно филтрирање и снимака project_video02 и challenge01) позивом enhance_contrast функције која помоћу методе прилагођеног хистограмског изједначавања не само да појачава контраст између трака и остатка пута, него га и ублажава у оним дијеловима траке гдје имамо смањено освјетљење (нпр. сијенка дрвета која прекрива траке у снимку challenge01.mp4 ) и тиме уклања испрекидане тачке и олакшава проналажење линија у наредним корацима. Након тога простим позивом cv2.threshold() и одрјеђивањем високог прага за шумове 215 (што смо у стању урадити због доста добре трансформације боја урађене у enhance_contrast) добијамо бинарну слику правимо бинарну слику која изгледа:

[image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Преобличавање слике у поглед на одозго ("bird's-eye view") је урађен кориштењем већ дате нам warper функције којој смо прослиједили слику цијелог погледа камере, и списак кључних тачака у тзв. "пољу од значаја" (points_of_interest) и координате нове слике на које их желимо мапирати (у нашем случају сами ћошкови слике са истом резолуцијом) - пошто се ови параметри не мијењају значајно ни на једном од снимака нити оквира (тј. слика у снимку) они су издвојени изван петље у линијама 159-163 а само преобличавање слике је извршено у 181 линији након што смо установили гдје је потребно поставити тачке које ће ограничавати "поље од значаја" са 4 исцртавања cv2.circle, примјер слике током избора тачака и након преобличавања приаказани су испод:

[image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Проналажење линија је урађено унутар функције lanes_detection (линија 33). На улазу у функцију добијамо већ трансформисану и обрађену бинарну слику (са тракама означеним бијелом бојом а осталим шумовима црном бојом). За одрјеђивање тј. детекцију средишта трака колосијека је кориштен метод "Sliding windows" који користећи хистограм проналази скокове вриједности од значаја (које су у већ филтрираној бинарној слици од раније означене бијелим пикселима) на самом дну слике (тј. непосриједно испред посматрача - у овом случају камере нашег возила), затим тражећи густо збијене вриједности од значаја изнад (даље) тражећи наредне тачке у даљини (у нашем случају 60 пиксела даље), уколико наша тренутна тачка нема одговарајућу вриједност, покушавамо пронаћи која од тачака у њеној непосриједној близини, било лијево или десно (m01, m10) има одговарајуће вриједности те помијерамо средиште наредног прозора (праваоугаоника) који ће означавати препознату линију. Њихове позиције одрјеђујемо полиномима другог степена у линији 82,83 користећи већ горе срачуната средишта трака.

[image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Закривљење пута као и положај кола у односу на средиште колосијека су мјерени и описани унутар засебне функције position_and_curvature_estimation (линија 87) она за параметре прима бинарну слику са препознатим тракама (бијело обојеним представљене траке и јачи шумови у ријетким случајевима када се појаве) и занемареним свим осталим дијеловима слике (обојеним црном бојом). Положај кола на колосијеку је одријеђен најприје одрјеђивањем средишта најближих кадрова слике (тј. половином ширине слике тј. по Х оси - у нашем случају 480, 540) затим, користећи претходно пронађена средишта тачака лијеве и десне траке нашег колосијека (left_centers, right_centers) одриједили смо најближе тачке почетка ових трака left_track_bottom и right_track_bottom које смо искористили за одређивање средишта колосијека простом аритметичком средином. Положај кола у односу на средину колосијека је одријеђен простим одузимањем гдје закључујемо да уколико је добијени број <0 ради се о томе да је положај средишта слике на "мањим координатама" од средитша колосијека тј. да је лијево од њега, а уколико је >0 да је десно од средишта колосијека, што смо и покушали приказати на крајњем излазном снимку.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Након извршене обраде свих података, приказ изворне слике изгледа овако:
[image6]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!! Записане снимке нисам могао покренути, те самим тиме ни окачити нигдје (јер нисам могао снимити датотеке), нисам успио установити тачан разлог овога.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Ово је био задатак са доста потешкоћа, прва од њих је свакако калибрација, која доста зависи од квалитета улаза (снимка са камере), лошије
снимке је теже обрадити јер је теже пронаћи разлике између битних података (у овом случају трака на цести) од небитних података (остатка 
цесте, њених закрпа, позадине, неравнина, трагова кочења и сл.). Апликација, иако добро уочава, прати и проналази линије колосијека у
којем се крећемо на project_video01, project_video02, project_video03 и challenge01, не успијева пронаћи те линије у случају одсјаја Сунца 
у камеру (challenge03), боје цесте која је доста слична тракама без претходно установљених већих разлика на ранијим дионицама (challenge02)
великим бројем кривина и осталим учесницима у саобраћају који се крећу у истој траци (моториста у challenge03) итд. Апликација такође није
у стању извршити подробније анализе података (као нпр. гдје се тачно налази колосијек, коју раздаљину мјеримо, положај кола унутар колосијека, степен закривљености колосијека тј. пута којим возимо) и приказати их у читљивом облику за кориснике усљед грешака у алгоритмима и њиховим прорачунима у каснијим корацима обраде које нисам успио отклонити на вријеме. Апликација у одријеђеној мјери јесте у стању препознати гдје се налази колосијек (тј. саставити ужи скуп од почетног у којем смо тражили траке нашег колосијека), али поприлично нетачно јер искорачава знатно изван трака које га означавају (као што се види на слици у 6. тачки) усљед неких грешака у прорачунима формула. Уз мање измјене, апликација би могла само издавати упозорења уколико траке колосијека нису уочене, у овим прорачунима се подједнако лоше сналази на сва 4 гореспоменута примјера на којима су траке успјешно издвојене и препознате. Препознавање трака бијаше доста лоше за challenge02 и challenge03 који се након израде корака 4. и 5. уопште нису покретали.


